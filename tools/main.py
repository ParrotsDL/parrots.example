import argparse
import random
import time
from addict import Dict
import yaml
import json
import socket

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.backends import cudnn
import torch.optim
from utils.data_loader import build_dataloader
from utils.save_util import Saver
from utils.scheduler import get_scheduler
from utils.optimizer import build_optimizer
from utils.meters import AverageMeter, ProgressMeter
from utils.misc import logger, accuracy, build_syncbn
import models
from utils.dist_util import DistributedModel, HalfModel
import utils.dist_util as dist
from SenseAgentClient import SenseAgentClientNG as sa


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--config',
    default='configs/resnet50.yaml',
    type=str,
    help='path to config file')
parser.add_argument(
    '--display',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--test',
    dest='test',
    action='store_true',
    help='evaluate model on validation set')

best_acc1 = 0.
local_rank = 0
rank = 0
global_size = 1


def main():
    args = parser.parse_args()
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg = Dict(args.config)

    global best_acc1, local_rank, rank, global_size

    rank, global_size, local_rank = dist.init()

    logger("=> rank {} of {} jobs, in {}".format(
        rank, global_size, socket.gethostname()), rank=-1)
    dist.barrier()
    logger("config file: \n{}".format(
        json.dumps(cfg, indent=2, ensure_ascii=False)))

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True

    logger("=> creating model '{}'".format(cfg.net.type))
    model = models.__dict__[cfg.net.type](**cfg.net.kwargs)
    model.cuda()

    if global_size > 1:
        args.dist = True
        model = DistributedModel(model)
    else:
        args.dist = False
    if cfg.net.syncbn == 1:
        logger("=> syncbn mode")
        model = build_syncbn(model, dist.group.WORLD)
    if cfg.trainer.get('mixed_training', False):
        model = HalfModel(model, cfg.trainer.get('float_layers', None))
        args.mixed_training = True
        logger("=> mix training mode")
    else:
        args.mixed_training = False

    # define loss function (criterion), optimizer, and lr_scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = build_optimizer(cfg.trainer, model)
    cudnn.benchmark = True

    args.start_epoch = -cfg.trainer.lr_scheduler.get('warmup_epochs', 0)
    args.max_epoch = cfg.trainer.max_epoch
    args.test_freq = cfg.trainer.test_freq
    args.log_freq = cfg.trainer.log_freq

    # optionally resume from a checkpoint
    use_resume = False
    cfg_saver = cfg.saver
    cfg_saver.checkpoint = None
    if cfg.saver.resume_model:
        use_resume = True
        cfg_saver.checkpoint = cfg_saver.resume_model
        logger('=> resume checkpoint "{}"'.format(cfg_saver.checkpoint))
    elif cfg_saver.pretrain_model:
        cfg_saver.checkpoint = cfg_saver.pretrain_model
        logger('=> load checkpoint "{}"'.format(cfg_saver.checkpoint))

    saver = Saver(cfg.net.type, cfg_saver.save_dir)
    if cfg_saver.checkpoint:
        saver.load_state(model, cfg_saver, strict=False)

    # generate a static sacli in main thread to enable dist cache
    if cfg.dataset.type == "senseagent":
        if cfg.dataset.senseagent_config.distcache == True:
            static_sacli = sa.SenseAgent(cfg.dataset.senseagent_config.userkey,
                                         cfg.dataset.senseagent_config.namespace,
                                         cfg.dataset.train.dataset_name,
                                         cfg.dataset.senseagent_config.user,
                                         cfg.dataset.senseagent_config.ip,
                                         cfg.dataset.senseagent_config.port)
            static_sacli.loadMetainfos(cfg.dataset.train.meta_source)
            my_rank = static_sacli.startDistCache(0.5)
            print("my rank is", my_rank)

    # Data loading code
    train_loader, train_sampler, test_loader, test_sampler = build_dataloader(
        cfg.dataset, dataset_type=cfg.dataset.type, total_epoch=args.max_epoch)

    # test mode
    if args.test:
        test(test_loader, model, criterion, args)
        return

    # restore optimizer
    if use_resume and saver.checkpoint:
        best_acc1, args.start_epoch = saver.load_optimizer(optimizer)

    # choose scheduler
    lr_scheduler = get_scheduler(optimizer, len(train_loader), args.max_epoch,
                                 cfg.trainer.lr_scheduler)

    # import monitor
    use_monitor = cfg.monitor
    monitor_writer = None
    if rank == 0 and use_monitor:
        if use_monitor.type == 'tensorboard':
            from tensorboardX import SummaryWriter
            monitor_writer = SummaryWriter(use_monitor.kwargs.logdir)
            cfg_saver.save_pavi = False
        elif use_monitor.type == 'pavi':
            from pavi import SummaryWriter
            monitor_writer = SummaryWriter(
                session_text=yaml.dump(args.config), **use_monitor.kwargs)
            if use_monitor.get("_taskid", None):
                monitor_writer._taskid = use_monitor._taskid

    # training
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch,
              args, monitor_writer)
        cur_iter = (epoch + 1) * len(train_loader)

        # save checkpoint
        if (epoch + 1) % cfg_saver.save_epoch_freq == 0 or epoch + 1 == args.max_epoch:
            # if cfg_saver.sync_bn_stats:
            #     model.average_bn_stats()
            if rank == 0:
                save_name = '{}_ckpt_e{}.pth'.format(cfg.net.type, epoch + 1)
                saver.save_ckpt(monitor_writer, epoch, cfg_saver, model,
                                optimizer, save_name, best_acc1, cur_iter)

        if args.test_freq < 0 or epoch < 0:
            continue
        if (epoch + 1) % args.test_freq == 0 or epoch + 1 == args.max_epoch:
            # evaluate on validation set
            loss, acc1, acc5 = test(test_loader, model, criterion, args)
            if rank == 0 and monitor_writer:
                monitor_writer.add_scalar('Accuracy', acc1, cur_iter)
                monitor_writer.add_scalar('Accuracy.top5', acc5, cur_iter)
                monitor_writer.add_scalar('Test.loss', loss, cur_iter)
            # remember best acc@1 and save checkpoint
            if acc1 > best_acc1:
                best_acc1 = acc1
                if rank == 0 and cfg_saver.save_best:
                    save_name = '{}_ckpt_best.pth'.format(cfg.net.type)
                    saver.save_ckpt(monitor_writer, epoch, cfg_saver, model,
                                    optimizer, save_name, best_acc1, cur_iter)


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args, monitor_writer):
    batch_time = AverageMeter('Time', ':6.3f', 200)
    data_time = AverageMeter('Data', ':6.3f', 200)
    losses = AverageMeter('Loss', ':6.4f', 1)
    top1 = AverageMeter('Acc@1', ':6.2f', 1)
    top5 = AverageMeter('Acc@5', ':6.2f', 1)
    cur_lr = AverageMeter('LR', ':6.4f', 1)
    memory = AverageMeter('Memory(MB)', ':.0f', 1)
    log_time = AverageMeter('', '', 1)
    progress = ProgressMeter(
        len(train_loader), log_time, batch_time, data_time, losses, top1, top5, cur_lr,
        memory, prefix="Epoch: [{}/{}] ".format(epoch + 1, args.max_epoch))

    # switch to train mode
    model.cuda().train()

    cur_iter = epoch * len(train_loader)
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        if args.mixed_training:
            input = input.half()
        target = target.cuda()
        lr_scheduler.step(cur_iter)
        cur_iter += 1

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item())
        top1.update(acc1[0])
        top5.update(acc5[0])
        cur_lr.update(lr_scheduler.get_lr()[0])
        memory.update(torch.cuda.max_memory_allocated()/1024/1024)
        log_time.update(time.strftime('%Y-%m-%d %H:%M:%S'))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.mixed_training:
            loss *= optimizer.loss_scale
        loss.backward()
        if args.dist:
            model.average_gradients()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0 or i % args.log_freq == 0:
            progress.print_log(i)
            if rank == 0 and monitor_writer:
                monitor_writer.add_scalar('Loss', losses.val, cur_iter)
                monitor_writer.add_scalar('Accuracy.train.top1', top1.val,
                                          cur_iter)
                monitor_writer.add_scalar('Accuracy.train.top5', top5.val,
                                          cur_iter)
                monitor_writer.add_scalar('LR', cur_lr.val, cur_iter)


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', 200)
    data_time = AverageMeter('Data', ':6.3f', 200)
    losses = AverageMeter('Loss', ':6.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader), batch_time, data_time, losses, top1, top5, prefix='Test: ')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            if args.mixed_training:
                input = input.half()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item())
            top1.update(acc1[0])
            top5.update(acc5[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                progress.print_log(i)
    logger(' * All Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5'
           ' {top5.avg:.3f}'.format(loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
