'''
    This script has two function. First, convert a quantize model to float model;
    Second, save the input and output of model in model inference.
    Only used for cambricon.
'''
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
from torch.utils import quantize
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
                    choices=['slurm', 'mpi'],
                    help='distributed backend')
parser.add_argument('--device',
                    type=str,
                    default="gpu",
                    choices=['mlu', 'gpu'],
                    help='card type, for camb pytorch use mlu')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--quant2float', dest='quant2float',
                    action='store_true',
                    help='flag to convert quantize model to float')
parser.add_argument('--saveInOut', dest='saveInOut',
                    action='store_true',
                    help='save input and output of single iter')

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')
args = parser.parse_args()

use_camb = False
if torch.__version__ == "parrots":
    from parrots.base import use_camb


def main():
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)

    backend = "cncl" if use_camb else "nccl"

    if args.launcher == 'slurm':
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        node_parts = re.findall('[0-9]+', node_list)[-4:]
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
    args.dist = args.world_size >= 1
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)

    dist.init_process_group(backend=backend)
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
        #if args.device == "mlu":
            #torch.cuda.manual_seed(cfgs.seed)
        cudnn.deterministic = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        #if args.device == "mlu":
            #torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    # Data loading code
    train_loader, train_sampler, test_loader, _ = build_dataloader(
        cfgs.dataset, args.world_size)

    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)
    
    if args.quantify:
        model = quantize.convert_to_adaptive_quantize(model, len(train_loader))
        if use_camb:
            model = model.to_memory_format(torch.channels_last)
        model = model.cuda()
    
    logger.info("creating model '{}'".format(cfgs.net.arch))

    if args.dist:
        model = DDP(model, device_ids=[args.local_rank])
    logger.info("model\n{}".format(model))

    if cfgs.get('label_smooth', None):
        criterion = LabelSmoothLoss(cfgs.trainer.label_smooth,
                                    cfgs.net.kwargs.num_classes).cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    #if args.device == "mlu":
        #criterion = criterion.to(ct.mlu_device())
    #else:
    criterion = criterion.cuda()
    logger.info("loss\n{}".format(criterion))

    optimizer = torch.optim.SGD(model.parameters(),
                                **cfgs.trainer.optimizer.kwargs)

    logger.info("optimizer\n{}".format(optimizer))

    cudnn.benchmark = True

    args.start_epoch = -cfgs.trainer.lr_scheduler.get('warmup_epochs', 0)
    args.max_epoch = cfgs.trainer.max_epoch
    args.test_freq = cfgs.trainer.test_freq
    args.log_freq = cfgs.trainer.log_freq
    args.lr = cfgs.trainer.optimizer.kwargs['lr']

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
    if not args.saveInOut:
        if args.quantify:
            logger.info("Executing the quantize model test ...")
            test(test_loader, model, criterion, args)
        else:
            logger.info("Executing the fix bitwidth model test ...")
            test(test_loader, model, criterion, args)
        time.sleep(5)

    if args.quant2float:
        logger.info("Converting quantize model to a float model ...")
        model = quantize.convert_from_adaptive_quantize(model)
        model = model.to_memory_format(torch.channels_last)
        model_path = cfgs.saver.pretrain_model
        test(test_loader, model, criterion, args, model_path)
        float_model_name = model_path.split('.')[0] + '_float.pth'
        torch.save(model.cpu().state_dict(), float_model_name)


def test(test_loader, model, criterion, args, model_path=None):
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
            if args.saveInOut and model_path != None:
                import numpy as np
                import sys
                if i == 0:
                    in_cpu = input.data.cpu().numpy()
                    out_cpu = output.data.cpu().numpy()
                    in_name = model_path.split('.')[0] + '_input.npy'
                    out_name = model_path.split('.')[0] + '_output.npy'
                    np.save(in_name, in_cpu)
                    np.save(out_name, out_cpu)
                    logger.info('input.size: {}'.format(input.size()))
                    logger.info('output.size: {}'.format(output.size()))
                    logger.info("Save input and output over, just stop! \n")
                    sys.exit()

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
            dist.all_reduce(loss.to(ct.mlu_device()))
        else:
            dist.all_reduce(loss.cuda())
        loss_avg = loss.item() / args.world_size
        if args.device == "mlu":
            dist.all_reduce(stats_all.to(ct.mlu_device()))
        else:
            dist.all_reduce(stats_all.cuda())
        acc1 = stats_all[0].item() * 100.0 / stats_all[2].item()
        acc5 = stats_all[1].item() * 100.0 / stats_all[2].item()

        logger.info(
            ' * All Loss {:.4f} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.
            format(loss_avg, acc1, stats_all[0].item(), stats_all[2].item(),
                   acc5, stats_all[1].item(), stats_all[2].item()))

    return loss_avg, acc1, acc5


if __name__ == '__main__':
    main()
