import os
import shutil
import argparse
import random
import time
import yaml
import json
import socket
import logging
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.backends import cudnn

import pape
import pape.distributed as dist
from pape.parallel import DistributedModel
from pape.half import HalfModel, HalfOptimizer
from pape.utils.lr_scheduler import build_scheduler
from pape.utils.loss import LabelSmoothLoss

import models
from utils.dataloader import build_dataloader
from utils.misc import accuracy, check_keys, AverageMeter, ProgressMeter
from utils.ema import EMA
from utils.optimizer import build_optimizer

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--config', default='configs/resnet50.yaml',
                    type=str, help='path to config file')
parser.add_argument('--test', dest='test', action='store_true',
                    help='evaluate model on validation set')

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')


def main():
    args = parser.parse_args()
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)

    args.rank, args.world_size, args.local_rank = dist.init()

    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)

    logger_all.info("rank {} of {} jobs, in {}".format(args.rank, args.world_size,
                    socket.gethostname()))

    dist.barrier()

    logger.info("config\n{}".format(json.dumps(cfgs, indent=2, ensure_ascii=False)))

    if cfgs.get('seed', None):
        random.seed(cfgs.seed)
        torch.manual_seed(cfgs.seed)
        torch.cuda.manual_seed(cfgs.seed)
        cudnn.deterministic = True

    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)
    model.cuda()

    logger.info("creating model '{}'".format(cfgs.net.arch))

    args.syncbn = False
    if cfgs.trainer.get('bn', None):
        if cfgs.trainer.bn.get('syncbn', False) is True:
            model = pape.utils.op.convert_syncbn(model)
            args.syncbn = True

    args.half = False
    if cfgs.trainer.get('mixed_precision', None):
        mix_cfg = cfgs.trainer.mixed_precision
        if mix_cfg.get('half', False) is True:
            model = HalfModel(model, float_bn=mix_cfg.get("float_bn", True),
                              float_module_type=eval(mix_cfg.get("float_module_type", "{}")),
                              float_module_name=eval(mix_cfg.get("float_module_name", "{}")))
            args.half = True

    model = DistributedModel(model)
    logger.info("model\n{}".format(model))

    if cfgs.trainer.get('label_smooth', None):
        criterion = LabelSmoothLoss(cfgs.trainer.label_smooth, cfgs.net.kwargs.num_classes).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    logger.info("loss\n{}".format(criterion))

    optimizer = build_optimizer(model, cfgs)
    if args.half:
        optimizer = HalfOptimizer(optimizer, loss_scale=cfgs.trainer.mixed_precision.loss_scale)
    logger.info("optimizer\n{}".format(optimizer))

    cudnn.benchmark = True

    args.start_iter = 0
    args.max_iter = cfgs.trainer.max_iter
    args.test_freq = cfgs.trainer.test_freq
    args.log_freq = cfgs.trainer.log_freq
    args.best_acc1 = 0.0

    # EMA
    if cfgs.trainer.ema.enable:
        cfgs.trainer.ema.kwargs.model = model
        ema = EMA(**cfgs.trainer.ema.kwargs)
    else:
        ema = None

    if cfgs.saver.resume_model:
        assert os.path.isfile(cfgs.saver.resume_model), 'Not found resume model: {}'.format(
            cfgs.saver.resume_model)
        checkpoint = torch.load(cfgs.saver.resume_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_iter = checkpoint['iter']
        args.max_iter -= args.start_iter
        args.best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'ema' in checkpoint.keys():
            ema.load_state_dict(checkpoint['ema'])
        logger.info("resume training from '{}' at iter {}".format(
            cfgs.saver.resume_model, checkpoint['iter']))
    elif cfgs.saver.pretrain_model:
        assert os.path.isfile(cfgs.saver.pretrain_model), 'Not found pretrain model: {}'.format(
            cfgs.saver.pretrain_model)
        checkpoint = torch.load(cfgs.saver.pretrain_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("pretrain training from '{}'".format(cfgs.saver.pretrain_model))

    if args.rank == 0 and cfgs.saver.get('save_dir', None):
        if not os.path.exists(cfgs.saver.save_dir):
            os.makedirs(cfgs.saver.save_dir)
            logger.info("create checkpoint folder {}".format(cfgs.saver.save_dir))

    # Data loading code
    train_loader, train_sampler, test_loader, _ = build_dataloader(cfgs.dataset, args.world_size, args.max_iter)

    # test mode
    if args.test:
        if ema is not None:
            ema.load_ema(model)
        test(test_loader, model, criterion, args)
        return

    # choose scheduler
    lr_cfg = cfgs.trainer.lr_scheduler
    lr_cfg.kwargs.optimizer = optimizer if isinstance(optimizer, torch.optim.Optimizer) else optimizer.optimizer
    lr_scheduler = build_scheduler(lr_cfg)

    monitor_writer = None
    if args.rank == 0 and cfgs.get('monitor', None):
        if cfgs.monitor.get('type', None) == 'pavi':
            from pavi import SummaryWriter
            if cfgs.monitor.get("_taskid", None):
                monitor_writer = SummaryWriter(
                    session_text=yaml.dump(args.config), **cfgs.monitor.kwargs, taskid=cfgs.monitor._taskid)
            else:
                monitor_writer = SummaryWriter(
                    session_text=yaml.dump(args.config), **cfgs.monitor.kwargs)

    # train max iter
    train(train_loader, test_loader, model, criterion, optimizer, lr_scheduler, args, cfgs, monitor_writer, ema)


def train(train_loader, test_loader, model, criterion, optimizer, lr_scheduler, args, cfgs, monitor_writer, ema):
    batch_time = AverageMeter('Time', ':.3f', 200)
    data_time = AverageMeter('Data', ':.3f', 200)

    losses = AverageMeter('Loss', ':.4f', 50)
    top1 = AverageMeter('Acc@1', ':.2f', 50)
    top5 = AverageMeter('Acc@5', ':.2f', 50)

    memory = AverageMeter('Memory(MB)', ':.0f')
    progress = ProgressMeter(args.max_iter, batch_time, data_time, losses, top1, top5,
                             memory, prefix="Iter: ")

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        cur_iter = args.start_iter + i
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        lr_scheduler.step(cur_iter)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        stats_all = torch.tensor([loss.item(), acc1[0].item(), acc5[0].item()]).float()
        dist.all_reduce(stats_all)
        stats_all /= args.world_size

        losses.update(stats_all[0].item())
        top1.update(stats_all[1].item())
        top5.update(stats_all[2].item())
        memory.update(torch.cuda.max_memory_allocated()/1024/1024)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.half:
            loss *= optimizer.loss_scale
        loss.backward()
        model.average_gradients()
        optimizer.step()

        if ema is not None:
            ema.step(model, curr_step=cur_iter)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cur_iter % args.log_freq == 0:
            progress.display(cur_iter)
            if args.rank == 0 and monitor_writer:
                monitor_writer.add_scalar('Train_Loss', losses.avg, cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top1', top1.avg, cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top5', top5.avg, cur_iter)

        if (cur_iter + 1) % args.test_freq == 0 or (cur_iter + 1) == args.max_iter:
            # evaluate on validation set
            val_loss, acc1, acc5 = test(test_loader, model, criterion, args)
            if ema is not None:
                ema.load_ema(model)
                if args.rank == 0:
                    print('============ EMA ============')
                val_loss_ema, acc1_ema, acc5_ema = test(test_loader, model, criterion, args)
                ema.recover(model)
            if args.rank == 0:
                if monitor_writer:
                    monitor_writer.add_scalar('Accuracy_Test_top1', acc1, cur_iter)
                    monitor_writer.add_scalar('Accuracy_Test_top5', acc5, cur_iter)
                    monitor_writer.add_scalar('Test_loss', val_loss, cur_iter)
                    if ema is not None:
                        monitor_writer.add_scalar('EMA_Accuracy_Test_top1', acc1_ema, cur_iter)
                        monitor_writer.add_scalar('EMA_Accuracy_Test_top5', acc5_ema, cur_iter)
                        monitor_writer.add_scalar('EMA_Test_loss', val_loss_ema, cur_iter)

                checkpoint = {
                    'iter': cur_iter,
                    'arch': cfgs.net.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': args.best_acc1,
                    'optimizer': optimizer.state_dict()
                }
                if ema is not None:
                    checkpoint['ema'] = ema.state_dict()

                ckpt_path = os.path.join(cfgs.saver.save_dir, cfgs.net.arch + '_ckpt_iter_{}.pth'.format(cur_iter))
                best_ckpt_path = os.path.join(cfgs.saver.save_dir, cfgs.net.arch + '_best.pth')
                torch.save(checkpoint, ckpt_path)
                if acc1 > args.best_acc1:
                    args.best_acc1 = acc1
                    shutil.copyfile(ckpt_path, best_ckpt_path)


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':.3f', 10)
    losses = AverageMeter('Loss', ':.4f', -1)
    top1 = AverageMeter('Acc@1', ':.2f', -1)
    top5 = AverageMeter('Acc@5', ':.2f', -1)
    stats_all = torch.Tensor([0, 0, 0]).long()
    progress = ProgressMeter(len(test_loader), batch_time, losses, top1, top5,
                             prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5), raw=True)

            losses.update(loss.item())
            top1.update(acc1[0].item() * 100.0 / target.size(0))
            top5.update(acc5[0].item() * 100.0 / target.size(0))

            stats_all.add_(torch.tensor([acc1[0].item(), acc5[0].item(), target.size(0)]).long())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)

        logger_all.info(' Rank {} Loss {:.4f} Acc@1 {} Acc@5 {} total_size {}'.format(
                        args.rank, losses.avg, stats_all[0].item(), stats_all[1].item(),
                        stats_all[2].item()))

        loss = torch.tensor([losses.avg])
        dist.all_reduce(loss)
        loss_avg = loss.item() / args.world_size
        dist.all_reduce(stats_all)
        acc1 = stats_all[0].item() * 100.0 / stats_all[2].item()
        acc5 = stats_all[1].item() * 100.0 / stats_all[2].item()

        logger.info(' * All Loss {:.4f} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.format(loss_avg,
                    acc1, stats_all[0].item(), stats_all[2].item(),
                    acc5, stats_all[1].item(), stats_all[2].item()))
    model.train()

    return loss_avg, acc1, acc5


if __name__ == '__main__':
    main()
