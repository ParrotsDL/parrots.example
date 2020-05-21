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
from torch.backends import cudnn

import pape
import pape.distributed as dist
from pape.parallel import DistributedModel
from pape.half import HalfModel, HalfOptimizer
from pape.utils.lr_scheduler import build_scheduler
from pape.utils.loss import LabelSmoothLoss

import models
from utils.dataloader import build_iter_dataloader, build_iter_dataloader_test
from utils.misc import accuracy, check_keys, AverageMeter, ProgressMeter, \
                       mixup_data, mixup_criterion

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--config', default='configs/resnet.yaml',
                    type=str, help='path to config file')
parser.add_argument('--test', dest='test', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--checkpoint', default='',
                    type=str, help='path to model checkpoint')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='resume train from checkpoint')

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')


def main():
    args = parser.parse_args()
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)

    args.rank, args.world_size, args.local_rank = dist.init()

    if args.test is False and cfgs.get('gpus_need', None):
        assert cfgs.gpus_need == args.world_size, 'Error: need gpu {}, but use gpu {}'.format(
                                                  cfgs.gpus_need, args.world_size)

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

    cudnn.benchmark = True

    logger.info("creating model '{}'".format(cfgs.net.arch))
    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)
    model.cuda()

    args.bn = 'bn'
    if cfgs.net.get('bn', None):
        bn_cfg = cfgs.net.bn
        args.bn = bn_cfg.type
        #if bn_cfg.type == 'syncbn':
        #    model = pape.utils.op.convert_syncbn(model, **bn_cfg.kwargs)

    args.half = False
    if cfgs.trainer.get('mixed_precision', None):
        mix_cfg = cfgs.trainer.mixed_precision
        if mix_cfg.type == 'half':
            args.half = True
            model = HalfModel(model, float_bn=mix_cfg.get("float_bn", True),
                              float_module_type=eval(mix_cfg.get("float_module_type", "{}")),
                              float_module_name=eval(mix_cfg.get("float_module_name", "{}")))

    model = DistributedModel(model)
    logger.info("model\n{}".format(model))

    if cfgs.trainer.get('label_smooth', None):
        criterion = LabelSmoothLoss(cfgs.trainer.label_smooth, cfgs.net.kwargs.num_classes)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    logger.info("loss\n{}".format(criterion))

    args.mixup = cfgs.trainer.get('mixup', None)
    logger.info("mixup: {}".format(args.mixup))

    optimizer = torch.optim.__dict__[cfgs.trainer.optimizer.type](
                    model.parameters(), **cfgs.trainer.optimizer.kwargs)
    if args.half:
        optimizer = HalfOptimizer(optimizer, loss_scale=mix_cfg.loss_scale)
    logger.info("optimizer\n{}".format(optimizer))

    args.start_iter = cfgs.trainer.start_iter
    args.end_iter = cfgs.trainer.end_iter
    args.test_freq = cfgs.trainer.test_freq
    args.log_freq = cfgs.trainer.log_freq
    args.best_acc1 = 0.0

    if args.checkpoint:
        assert os.path.isfile(args.checkpoint), 'Not found checkpoint: {}'.format(
            args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        if args.resume:
            args.start_iter = checkpoint['last_iter'] + 1
            args.best_acc1 = checkpoint['best_acc1']
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("resume training from '{}' at iter {}".format(
                args.checkpoint, args.start_iter))
        else:
            logger.info("load checkpoint from '{}'".format(args.checkpoint))

    args.save_dir = cfgs.trainer.get('save_dir', None)
    if args.rank == 0 and args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            logger.info("create checkpoint folder {}".format(args.save_dir))

    # build dataloader
    train_loader, train_sampler, test_loader, _, _, test_dataset = build_iter_dataloader(cfgs.data, args.world_size, args.end_iter - args.start_iter + 1)

    args.test_dataset = test_dataset

    # test mode
    if args.test:
        test(test_loader, model, criterion, args, cfgs)
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
    train(train_loader, test_loader, model, criterion, optimizer, lr_scheduler, args, cfgs, monitor_writer)


def train(train_loader, test_loader, model, criterion, optimizer, lr_scheduler, args, cfgs, monitor_writer):
    batch_time = AverageMeter('Time', ':.3f', 200)
    data_time = AverageMeter('Data', ':.3f', 200)

    losses = AverageMeter('Loss', ':.4f', 50)
    top1 = AverageMeter('Acc@1', ':.2f', 50)
    top5 = AverageMeter('Acc@5', ':.2f', 50)

    memory = AverageMeter('Memory(MB)', ':.0f')
    lr = AverageMeter('LR', ':.6f')
    progress = ProgressMeter(args.end_iter, batch_time, data_time, losses, top1, top5,
                             memory, lr, prefix="Iter: ")

    # switch to train mode
    model.train()
    args.cur_iter = args.start_iter
    end = time.time()
    for input, target in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.step(args.cur_iter)
        lr.update(lr_scheduler.get_lr()[0])

        input = input.cuda()
        target = target.cuda()

        if args.mixup:
            mixed_input, target_a, target_b, lam = mixup_data(input, target, args.mixup)
            output = model(mixed_input)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
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
            loss = optimizer.scale_up_loss(loss)
        loss.backward()
        model.average_gradients()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.cur_iter % args.log_freq == 0:
            progress.display(args.cur_iter)
            if args.rank == 0 and monitor_writer:
                monitor_writer.add_scalar('Train_Loss', losses.avg, args.cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top1', top1.avg, args.cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top5', top5.avg, args.cur_iter)

        if args.cur_iter % args.test_freq == 0 or args.cur_iter == args.end_iter:
            # evaluate on validation set
            #val_loss, acc1, acc5 = test(test_loader, model, criterion, args, cfgs)
            val_loss, acc1, acc5 = 1, 1, 1

            if args.rank == 0:
                if monitor_writer:
                    monitor_writer.add_scalar('Accuracy_Test_top1', acc1, args.cur_iter)
                    monitor_writer.add_scalar('Accuracy_Test_top5', acc5, args.cur_iter)
                    monitor_writer.add_scalar('Test_loss', val_loss, args.cur_iter)

                checkpoint = {
                    'last_iter': args.cur_iter,
                    'arch': cfgs.net.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': args.best_acc1,
                    'optimizer': optimizer.state_dict()
                }

                ckpt_path = os.path.join(args.save_dir, cfgs.net.arch + '_iter_{}.pth'.format(args.cur_iter))
                best_ckpt_path = os.path.join(args.save_dir, cfgs.net.arch + '_best.pth')
                torch.save(checkpoint, ckpt_path)
                if acc1 > args.best_acc1:
                    args.best_acc1 = acc1
                    shutil.copyfile(ckpt_path, best_ckpt_path)
        args.cur_iter += 1


def test(test_loader, model, criterion, args, cfgs):
    batch_time = AverageMeter('Time', ':.3f', 10)
    losses = AverageMeter('Loss', ':.4f', -1)
    top1 = AverageMeter('Acc@1', ':.2f', -1)
    top5 = AverageMeter('Acc@5', ':.2f', -1)
    stats_all = torch.Tensor([0, 0, 0]).long()
    progress = ProgressMeter(len(test_loader), batch_time, losses, top1, top5,
                             prefix="Test: ")

    test_loader, _, = build_iter_dataloader_test(cfgs.data)
    # switch to evaluate mode
    model.broadcast_model()
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

    del test_loader
    return loss_avg, acc1, acc5


if __name__ == '__main__':
    main()
