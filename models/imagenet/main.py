import os
import shutil
import argparse
import random
import re
import time
import yaml
import json
import socket
import logging
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.backends import cudnn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import models
from utils.dataloader import build_dataloader
from utils.misc import accuracy, check_keys, AverageMeter, ProgressMeter
from utils.loss import LabelSmoothLoss
from utils.lr_scheduler import adjust_learning_rate_cos

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--config',
                    default='configs/resnet50.yaml',
                    type=str,
                    help='path to config file')
parser.add_argument('--test',
                    dest='test',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--quantify',
                    dest='quantify',
                    action='store_true',
                    help='quantify training')
parser.add_argument('--port',
                    default=12345,
                    type=int,
                    metavar='P',
                    help='master port')
parser.add_argument('--dummy_test',
                    dest='dummy_test',
                    action='store_true',
                    help='dummy data for speed evaluation')
parser.add_argument('--launcher',
                    type=str,
                    default="slurm",
                    choices=['slurm', 'mpi', 'default'],
                    help='distributed backend')
parser.add_argument('--device',
                    type=str,
                    default="gpu",
                    choices=['mlu', 'gpu'],
                    help='card type, for camb pytorch use mlu')
parser.add_argument('--seed', type=int, default=None, help='random seed')

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')
args = parser.parse_args()

if args.device == "mlu":
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    import torch_mlu.core.mlu_quantize as qt
    import torch_mlu.core.device.notifier as Notifier
    import torch_mlu.core.quantized.functional as func
    ct.set_cnml_enabled(False)

use_camb = False
if torch.__version__ == "parrots":
    from parrots.base import use_camb


def main():
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)

    backend = "cncl" if use_camb or args.device == "mlu" else "nccl"

    if args.launcher == 'slurm':
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        node_parts = re.findall('[0-9]+', node_list)
        os.environ[
            'MASTER_ADDR'] = f'{node_parts[0]}.{node_parts[1]}.{node_parts[2]}.{node_parts[3]}'
        os.environ['MASTER_PORT'] = str(args.port)
    elif args.launcher == "mpi":
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    args.dist = args.world_size > 1
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)

    kwargs_name = "card_" + str(args.world_size)
    if kwargs_name in cfgs.trainer.optimizer.kwargs:
        hyper_kwargs = cfgs.trainer.optimizer.kwargs[kwargs_name]
    else:
        hyper_kwargs = cfgs.trainer.optimizer.kwargs.card_8

    dist.init_process_group(backend=backend)
    if args.device == "mlu":
        ct.set_device(args.local_rank)
    else:
        torch.cuda.set_device(args.local_rank)

    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)

    logger_all.info("rank {} of {} jobs, in {}".format(args.rank,
                                                       args.world_size,
                                                       socket.gethostname()))

    logger.info("config\n{}".format(
        json.dumps(cfgs, indent=2, ensure_ascii=False)))

    if cfgs.get('seed', None):
        random.seed(cfgs.seed)
        torch.manual_seed(cfgs.seed)
        if args.device == "mlu":
            torch.cuda.manual_seed(cfgs.seed)
        cudnn.deterministic = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "mlu":
            torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    # Data loading code
    train_loader, train_sampler, test_loader, _ = build_dataloader(
        cfgs.dataset, args.world_size)

    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)

    if args.device == "gpu":
        if args.quantify:
            from torch.utils import quantize
            model = quantize.convert_to_adaptive_quantize(
                model, len(train_loader))
        if use_camb:
            model = model.to_memory_format(torch.channels_last)
        model.cuda()
    elif args.device == "mlu":
        if args.quantify:
            model = qt.adaptive_quantize(model, len(train_loader))
        model = model.to(ct.mlu_device())

    logger.info("creating model '{}'".format(cfgs.net.arch))

    if args.dist:
        model = DDP(model, device_ids=[args.local_rank])
    logger.info("model\n{}".format(model))

    if cfgs.get('label_smooth', None):
        criterion = LabelSmoothLoss(cfgs.trainer.label_smooth,
                                    cfgs.net.kwargs.num_classes).cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    if args.device == "mlu":
        criterion = criterion.to(ct.mlu_device())
    else:
        criterion = criterion.cuda()
    logger.info("loss\n{}".format(criterion))

    optimizer = torch.optim.SGD(model.parameters(),
                                **hyper_kwargs)

    logger.info("optimizer\n{}".format(optimizer))

    cudnn.benchmark = True

    args.start_epoch = -cfgs.trainer.lr_scheduler.get('warmup_epochs', 0)
    args.max_epoch = cfgs.trainer.max_epoch
    args.test_freq = cfgs.trainer.test_freq
    args.log_freq = cfgs.trainer.log_freq

    best_acc1 = 0.0
    if cfgs.saver.resume_model:
        assert os.path.isfile(
            cfgs.saver.resume_model), 'Not found resume model: {}'.format(
                cfgs.saver.resume_model)
        checkpoint = torch.load(cfgs.saver.resume_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("resume training from '{}' at epoch {}".format(
            cfgs.saver.resume_model, checkpoint['epoch']))
    elif cfgs.saver.pretrain_model:
        assert os.path.isfile(
            cfgs.saver.pretrain_model), 'Not found pretrain model: {}'.format(
                cfgs.saver.pretrain_model)
        checkpoint = torch.load(cfgs.saver.pretrain_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("pretrain training from '{}'".format(
            cfgs.saver.pretrain_model))

    if args.rank == 0 and cfgs.saver.get('save_dir', None):
        if not os.path.exists(cfgs.saver.save_dir):
            os.makedirs(cfgs.saver.save_dir)
            logger.info("create checkpoint folder {}".format(
                cfgs.saver.save_dir))

    args.arch = cfgs.net.arch

    # test mode
    if args.test:
        test(test_loader, model, criterion, args)
        return

    # choose scheduler
    lr_scheduler = torch.optim.lr_scheduler.__dict__[
        cfgs.trainer.lr_scheduler.type](optimizer if isinstance(
            optimizer, torch.optim.Optimizer) else optimizer.optimizer,
                                        **cfgs.trainer.lr_scheduler.kwargs,
                                        last_epoch=args.start_epoch - 1)

    # required accuracy
    req_acc = cfgs.trainer.req_acc if 'req_acc' in cfgs.trainer else 100

    # training
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch + 1) % args.test_freq == 0 or epoch + 1 == args.max_epoch:
            # evaluate on validation set
            loss, acc1, acc5 = test(test_loader, model, criterion, args)

            if args.rank == 0:

                checkpoint = {
                    'epoch': epoch + 1,
                    'arch': cfgs.net.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict()
                }

                ckpt_path = os.path.join(
                    cfgs.saver.save_dir,
                    cfgs.net.arch + '_ckpt_epoch_{}.pth'.format(epoch))
                best_ckpt_path = os.path.join(cfgs.saver.save_dir,
                                              cfgs.net.arch + '_best.pth')
                torch.save(checkpoint, ckpt_path)
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    shutil.copyfile(ckpt_path, best_ckpt_path)
                    
            if acc1 >= req_acc:
                logger.info(
                    "Reach required accuracy {}. Current accuracy {:.2f} Training stops."
                    .format(req_acc, acc1))
                break

        if args.arch not in ["mobile_v2"]:
            lr_scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':.3f', 200)
    data_time = AverageMeter('Data', ':.3f', 200)

    losses = AverageMeter('Loss', ':.4f', 50)
    top1 = AverageMeter('Acc@1', ':.2f', 50)
    top5 = AverageMeter('Acc@5', ':.2f', 50)

    memory = AverageMeter('Memory(MB)', ':.0f')
    progress = ProgressMeter(len(train_loader),
                             batch_time,
                             data_time,
                             losses,
                             top1,
                             top5,
                             memory,
                             prefix="Epoch: [{}/{}]".format(
                                 epoch + 1, args.max_epoch))

    # switch to train mode
    model.train()
    end = time.time()
    if args.dummy_test:
        input_, target_ = next(iter(train_loader))
        train_loader = [(i, i) for i in range(len(train_loader))].__iter__()
        if args.device == "mlu":
            input_ = input_.to(ct.mlu_device())
            target_ = target_.to(ct.mlu_device())
        elif use_camb:
            input_ = input_.contiguous(torch.channels_last).cuda()
            target_ = target_.int().cuda()
        else:
            input_ = input_.cuda()
            target_ = target_.cuda()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if not args.dummy_test and args.arch == "mobile_v2":
            adjust_learning_rate_cos(optimizer, epoch, i, len(train_loader),
                                     args)
        if args.dummy_test:
            input = input_.detach()
            input.requires_grad = True
            target = target_
        elif args.device == "mlu":
            input = input.to(ct.mlu_device())
            target = target.to(ct.mlu_device())
        elif use_camb:
            input = input.contiguous(torch.channels_last).cuda()
            target = target.int().cuda()
        else:
            input = input.cuda()
            target = target.cuda()

        # compute output
        if args.arch == 'googlenet':
            aux1, aux2, output = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux1, target)
            loss3 = criterion(aux2, target)
            loss = loss1 + 0.3 * (loss2 + loss3)
        else:
            output = model(input)
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.device == "mlu":
            stats_all = torch.tensor(
                [loss.item(), acc1[0].item(),
                 acc5[0].item()]).float().to(ct.mlu_device())
        else:
            stats_all = torch.tensor(
                [loss.item(), acc1[0].item(), acc5[0].item()]).float().cuda()
        if args.dist:
            dist.all_reduce(stats_all)
        stats_all /= args.world_size

        losses.update(stats_all[0].item())
        top1.update(stats_all[1].item())
        top5.update(stats_all[2].item())
        if args.device == "gpu":
            memory.update(torch.cuda.max_memory_allocated() / 1024 / 1024)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_freq == 0:
            progress.display(i)


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':.3f', 10)
    losses = AverageMeter('Loss', ':.4f', -1)
    top1 = AverageMeter('Acc@1', ':.2f', -1)
    top5 = AverageMeter('Acc@5', ':.2f', -1)
    stats_all = torch.Tensor([0, 0, 0]).float()
    progress = ProgressMeter(len(test_loader),
                             batch_time,
                             losses,
                             top1,
                             top5,
                             prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            if args.device == "mlu":
                input = input.to(ct.mlu_device())
                target = target.to(ct.mlu_device())
            else:
                if use_camb:
                    input = input.contiguous(torch.channels_last).cuda()
                    target = target.int().cuda()
                else:
                    input = input.cuda()
                    target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5), raw=True)

            losses.update(loss.item())
            top1.update(acc1[0].item() * 100.0 / target.size(0))
            top5.update(acc5[0].item() * 100.0 / target.size(0))

            stats_all.add_(
                torch.tensor([acc1[0].item(), acc5[0].item(),
                              target.size(0)]).float())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)

        logger_all.info(
            ' Rank {} Loss {:.4f} Acc@1 {} Acc@5 {} total_size {}'.format(
                args.rank, losses.avg, stats_all[0].item(),
                stats_all[1].item(), stats_all[2].item()))

        loss = torch.tensor([losses.avg])
        if args.device == "mlu":
            loss.to(ct.mlu_device())
            stats_all.to(ct.mlu_device())
        else:
            loss.cuda()
            stats_all.cuda()
        dist.all_reduce(loss)
        dist.all_reduce(stats_all)
        
        loss_avg = loss.item() / args.world_size
        acc1 = stats_all[0].item() * 100.0 / stats_all[2].item()
        acc5 = stats_all[1].item() * 100.0 / stats_all[2].item()

        logger.info(
            ' * All Loss {:.4f} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.
            format(loss_avg, acc1, stats_all[0].item(), stats_all[2].item(),
                   acc5, stats_all[1].item(), stats_all[2].item()))

    return loss_avg, acc1, acc5


if __name__ == '__main__':
    main()
