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
import numpy as np
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
from torch.backends import cudnn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import models
from utils.dataloader import build_dataloader
from utils.misc import accuracy, check_keys, AverageMeter, ProgressMeter
from utils.loss import LabelSmoothLoss
from utils.dist_utils import DistributedModel
from pape.half.half_model import HalfModel
from pape.half.half_optimizer import HalfOptimizer


parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--config', default='configs/resnet50.yaml',
                    type=str, help='path to config file')
parser.add_argument('--test', dest='test', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dummy_test', dest='dummy_test', action='store_true',
                    help='dummy data for speed evaluation')
parser.add_argument('--pavi', dest='pavi', action='store_true', default=False, help='pavi use')
parser.add_argument('--pavi-project', type=str, default="default", help='pavi project name')
parser.add_argument('--max_step', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--taskid', default='None', type=str, help='pavi taskid')
parser.add_argument('--data_reader', type=str, default="MemcachedReader", choices=['MemcachedReader', 'CephReader'], help='io backend')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--port', default=12345, type=int, metavar='P',
                    help='master port')
parser.add_argument('--local_rank', type=int, default=0, help='cude device id')
parser.add_argument('--mixed', type=bool, default=True, help='if use mixed train')

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')


def main():
    start_time = time.time()
    iter_time_list = []
    args = parser.parse_args()
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)

    if not os.path.exists(cfgs.saver.save_dir):
        os.makedirs(cfgs.saver.save_dir)
    save_dir = cfgs.saver.save_dir
    
    # 创建 handler， 用于写入日志
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_file = os.path.join(save_dir, cfgs.net.arch + "_" + rq + ".log")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    # 定义 handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 将logger添加到handler里面
    logger.addHandler(fh)

    if 'SLURM_PROCID' in os.environ.keys():
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        node_parts = re.findall('[0-9]+', node_list)
        os.environ['MASTER_ADDR'] = f'{node_parts[1]}.{node_parts[2]}.{node_parts[3]}.{node_parts[4]}'
        os.environ['MASTER_PORT'] = str(args.port)
    elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ.keys():
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
    args.dist = args.world_size > 1
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    
    dist.init_process_group(backend="cncl")
    torch.cuda.set_device(args.local_rank) 

    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)

    logger_all.info("rank {} of {} jobs, in {}".format(args.rank, args.world_size,
                    socket.gethostname()))

    dist.barrier()
 
    logger.info("config\n{}".format(json.dumps(cfgs, indent=2, ensure_ascii=False)))


    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)
    #model = model.to_memory_format(torch.channels_last)
    #model.cuda()

    logger.info("creating model '{}'".format(cfgs.net.arch))
    #if args.dist:
    #    model = DistributedModel(model)
    logger.info("model\n{}".format(model))

    if cfgs.get('label_smooth', None):
        criterion = LabelSmoothLoss(cfgs.trainer.label_smooth, cfgs.net.kwargs.num_classes).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    logger.info("loss\n{}".format(criterion))

    optimizer = torch.optim.SGD(model.parameters(), **cfgs.trainer.optimizer.kwargs)
    logger.info("optimizer\n{}".format(optimizer))


    args.start_epoch = -cfgs.trainer.lr_scheduler.get('warmup_epochs', 0)
    args.max_epoch = cfgs.trainer.max_epoch
    if args.max_step is not None:
        args.max_epoch = args.max_step
    args.test_freq = cfgs.trainer.test_freq
    args.log_freq = cfgs.trainer.log_freq

    best_acc1 = 0.0
    if cfgs.saver.resume_model:
        assert os.path.isfile(cfgs.saver.resume_model), 'Not found resume model: {}'.format(
            cfgs.saver.resume_model)
        checkpoint = torch.load(cfgs.saver.resume_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.taskid = checkpoint['taskid']
        logger.info("resume training from '{}' at epoch {}".format(
            cfgs.saver.resume_model, checkpoint['epoch']))
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
    
    if args.mixed == True:
        model = HalfModel(model)

    model = model.to_memory_format(torch.channels_last)
    model = model.cuda()
    if args.dist:
         model = DistributedModel(model)
    # Data loading code
    train_loader, train_sampler, test_loader, _ = build_dataloader(cfgs.dataset, args.world_size, args.data_reader)

    # test mode
    if args.test:
        test(test_loader, model, criterion, args)
        return

    # choose scheduler
    lr_scheduler = torch.optim.lr_scheduler.__dict__[cfgs.trainer.lr_scheduler.type](
                       optimizer if isinstance(optimizer, torch.optim.Optimizer) else optimizer.optimizer,
                       **cfgs.trainer.lr_scheduler.kwargs, last_epoch=args.start_epoch - 1)

    monitor_writer = None
    
    if args.rank == 0 and (cfgs.get('monitor', None) or args.pavi):
       # if cfgs.monitor.get('type', None) == 'pavi':
        if args.pavi:
            monitor_kwargs = {'task': cfgs.net.arch, 'project': args.pavi_project}
        else:
            monitor_kwargs = cfgs.monitor.kwargs
            if hasattr(args, 'taskid'):
                monitor_kwargs['taskid'] = args.taskid
            elif hasattr(cfgs.monitor, '_taskid'):
                monitor_kwargs['taskid'] = cfgs.monitor._taskid
        from pavi import SummaryWriter
        monitor_writer = SummaryWriter(
            session_text=yaml.dump(args.config), **monitor_kwargs)
        args.taskid = monitor_writer.taskid

    run_time = time.time()

    # training
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, monitor_writer, iter_time_list)
        
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                       dtype=torch.int,
                       device=torch.device('cuda'))
        #if  args.world_size > 1:
        #    dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)

        mem_alloc = mem_mb.item()
        # get max memory cached
        mem = torch.cuda.max_memory_cached()
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                       dtype=torch.int,
                       device=torch.device('cuda'))
        #if args.world_size > 1:
        #    dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        mem_cached = mem_mb.item()
        
        if (epoch + 1) % args.test_freq == 0 or epoch + 1 == args.max_epoch:
            # evaluate on validation set
            loss, acc1, acc5 = test(test_loader, model, criterion, args)
            model = model.cpu().to_memory_format(torch.contiguous_format)
            if args.rank == 0:
                if monitor_writer:
                    monitor_writer.add_scalar('Accuracy_Test_top1', acc1, len(train_loader)*epoch)
                    monitor_writer.add_scalar('Accuracy_Test_top5', acc5, len(train_loader)*epoch)
                    monitor_writer.add_scalar('Test_loss', loss, len(train_loader)*epoch)

                checkpoint = {
                    'epoch': epoch + 1,
                    'arch': cfgs.net.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'taskid': args.taskid
                }

                ckpt_path = os.path.join(cfgs.saver.save_dir, cfgs.net.arch + '_ckpt_epoch_{}.pth'.format(epoch))
                best_ckpt_path = os.path.join(cfgs.saver.save_dir, cfgs.net.arch + '_best.pth')
                torch.save(checkpoint, ckpt_path)
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    shutil.copyfile(ckpt_path, best_ckpt_path)

        lr_scheduler.step()
    end_time = time.time()
    if args.rank == 0:
        logger.info('__benchmark_total_time(h): {}'.format((end_time - start_time) / 3600))
        logger.info('__benchmark_pure_training_time(h): {}'.format((end_time - run_time) / 3600))
        logger.info('__benchmark_avg_iter_time(s): {}'.format(np.mean(iter_time_list)))
        logger.info('__benchmark_mem_alloc(mb): {}'.format(mem_alloc))
        logger.info('__benchmark_mem_cached(mb): {}'.format(mem_cached))

    if args.rank == 0 and monitor_writer:
        monitor_writer.add_scalar('__benchmark_total_time(h)',(end_time - start_time) / 3600,1)
        monitor_writer.add_scalar('__benchmark_pure_training_time(h)',(end_time - run_time) / 3600,1)
        monitor_writer.add_scalar('__benchmark_avg_iter_time(s)',np.mean(iter_time_list),1)
        monitor_writer.add_scalar('__benchmark_mem_alloc(mb)',mem_alloc,1)
        monitor_writer.add_scalar('__benchmark_mem_cached(mb)',mem_cached,1)
        monitor_writer.add_snapshot('__benchmark_pseudo_snapshot', None, 1)

def train(train_loader, model, criterion, optimizer, epoch, args, monitor_writer, iter_time_list):
    batch_time = AverageMeter('Time', ':.3f', 200)
    data_time = AverageMeter('Data', ':.3f', 200)

    losses = AverageMeter('Loss', ':.4f', 50)
    top1 = AverageMeter('Acc@1', ':.2f', 50)
    top5 = AverageMeter('Acc@5', ':.2f', 50)

    memory = AverageMeter('Memory(MB)', ':.0f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, memory, prefix="Epoch: [{}/{}]".format(epoch + 1, args.max_epoch))

    # switch to train mode
    model.train()
    end = time.time()
    loader_length = len(train_loader)
    if args.dummy_test:
        input_, target_  = next(iter(train_loader))
        train_loader = [(i, i) for i in range(len(train_loader))].__iter__()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.dummy_test:
            input = input_.detach()
            input.requires_grad = True
            target = target_
        if args.mixed == True:
            input = input.half()
        input = input.contiguous(torch.channels_last)
        input = input.cuda()
        target = target.int().cuda()
        # compute output
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        acc5 = acc1.clone()
        stats_all = torch.tensor([loss.item(), acc1[0].item(), acc5[0].item()]).float().cuda()
        # pdb.set_trace()
        dist.all_reduce(stats_all)
        stats_all /= args.world_size

        losses.update(stats_all[0].item())
        top1.update(stats_all[1].item())
        top5.update(stats_all[2].item())
        memory.update(torch.cuda.max_memory_allocated()/1024/1024)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.dist:
            model.average_gradients()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        iter_end_time = time.time()
        if len(iter_time_list) <= 200 and i >= 800 and i <= 1000:
            iter_time_list.append(iter_end_time - iter_start_time)
            
        iter_start_time = time.time()
        if i % args.log_freq == 0:
            progress.display(i)
            if args.rank == 0 and monitor_writer:
                cur_iter = epoch * loader_length + i
                monitor_writer.add_scalar('Train_Loss', losses.avg, cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top1', top1.avg, cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top5', top5.avg, cur_iter)
        if os.environ.get('PARROTS_BENCHMARK') == '1' and i == 1010:
            return


def test(test_loader, model, criterion, args):
    logger = logging.getLogger()
    logger_all = logging.getLogger('all')
    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)
    batch_time = AverageMeter('Time', ':.3f', 10)
    losses = AverageMeter('Loss', ':.4f', -1)
    top1 = AverageMeter('Acc@1', ':.2f', -1)
    top5 = AverageMeter('Acc@5', ':.2f', -1)
    stats_all = torch.Tensor([0, 0, 0]).long()
    progress = ProgressMeter(len(test_loader), batch_time, losses, top1, top5,
                             prefix="Test: ")

    # switch to evaluate mode
    #model.eval()
    if True:
    #with torch.no_grad():
        end = time.time()
        if args.dummy_test:
            input_, target_ = next(iter(test_loader))
            test_loader = [(i, i) for i in range(len(test_loader))]
        for i, (input, target) in enumerate(test_loader):
            if args.dummy_test:
                input = input_
                target = target_
            if args.mixed == True:
                input = input.half()
            input = input.contiguous(torch.channels_last)
            input = input.cuda()
            target = target.int().cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,), raw=True)
            acc5 = acc1.clone()
            losses.update(loss.item())
            top1.update(acc1[0].item() * 100.0 / target.size(0))
            top5.update(acc5[0].item() * 100.0 / target.size(0))

            stats_all.add_(torch.tensor([acc1[0].item(), acc5[0].item(), target.size(0)]))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)

        logger_all.info(' Rank {} Loss {:.4f} Acc@1 {} Acc@5 {} total_size {}'.format(
                        args.rank, losses.avg, stats_all[0].item(), stats_all[1].item(),
                        stats_all[2].item()))

        loss = torch.tensor([losses.avg])
        loss = loss.cuda()
        dist.all_reduce(loss)
        loss_avg = loss.item() / args.world_size
        stats_all = stats_all.float().cuda()
        dist.all_reduce(stats_all)
        acc1 = stats_all[0].item() * 100.0 / stats_all[2].item()
        acc5 = stats_all[1].item() * 100.0 / stats_all[2].item()

        logger.info(' * All Loss {:.4f} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.format(loss_avg,
                    acc1, stats_all[0].item(), stats_all[2].item(),
                    acc5, stats_all[1].item(), stats_all[2].item()))

    return loss_avg, acc1, acc5


if __name__ == '__main__':
    main()