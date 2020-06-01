# flake8: noqa
import logging
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends as backends
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import models
import pape
import pape.distributed as dist
from pape.data import McDataset
from pape.utils.op import convert_syncbn
import parrots
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Benchmark PAPE parallel Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# pape args
parser.add_argument('--parallel', default='overlap',
                    choices=['overlap', 'nonoverlap'],
                    help='pape parallel type')
parser.add_argument('--bucket_size', default=1., type=float,
                    help='pape bucket_size(MB)')
parser.add_argument('--half', action='store_true', help='use pape half')
parser.add_argument('--loss_scale', default='dynamic', type=str,
                    help='pape half loss scale')
parser.add_argument('--syncbn', action='store_true', help='use pape syncbn')

# benchmark args
parser.add_argument('--benchmark', action='store_true',
                    help='benchmark mode, will use dummy input data')
parser.add_argument('--max_iter', default=2000, type=int,
                    help='max benchmark iteration')


def main():
    args = parser.parse_args()
    args.rank, args.world_size, args.local_rank = dist.init()

    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model.cuda()

    if args.syncbn:
        model = convert_syncbn(model)

    if args.half:
        model = pape.half.HalfModel(model)

    if args.parallel == "overlap":
        model = pape.parallel.DistributedModel(model, bucket_cap_mb=args.bucket_size)
    elif args.parallel == "nonoverlap":
        model = pape.parallel.DistributedModel(model, require_backward_overlap=False)

    if args.rank == 0:
        print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.rank == 0:
        print(criterion)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.half:
        if args.loss_scale != 'dynamic':
            args.loss_scale = float(args.loss_scale)
        optimizer = pape.half.HalfOptimizer(optimizer, loss_scale=args.loss_scale)


    backends.cudnn.benchmark = True


    train(model, criterion, optimizer, args)


def train(model, criterion, optimizer, args):
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    losses = AverageMeter('Loss', ':.4f', 10)
    top1 = AverageMeter('Acc@1', ':6.2f', 10)
    top5 = AverageMeter('Acc@5', ':6.2f', 10)

    if args.benchmark:
        dummy_input = torch.rand([args.batch_size, 3, 224, 224]).cuda()
        dummy_target = torch.randint(1000, (args.batch_size,), dtype=torch.long).cuda()
        max_iter = args.max_iter
        progress = ProgressMeter(max_iter, batch_time,losses, prefix="Benchmark: ")

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(max_iter):
        if i in range(0, 1000):
            optimizer.zero_grad()
            if args.benchmark:
                input, target = dummy_input, dummy_target
            # else:
            #     input, target = train_iter.next()

            input = input.cuda()
            target = target.cuda()
                
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            # if args.half:
            #     loss = optimizer.scale_up_loss(loss)
            loss.backward()

        if args.rank == 1 and i == 1002:
            print('skip forward&backward from now on, alexnet前向后向时间为0.03-0.04s')

        # allreduce
        # losses.update(loss.item())
        # input.cpu()
        model.average_gradients()
        
        # The following code is used to measure the impact of optimizer.step 
        if i in range(0,3000):
            optimizer.step()
        # if i in range(1000,2000):
        #     if args.rank == 1 and i == 1001:
        #         print('set all param.grad = None from now on')
        #     for param in model.parameters():
        #         if param.requires_grad:
        #             param.grad = None
        #     optimizer.step()
        # if i in range(2000,3000):
        #     if args.rank == 1 and i == 2001:
        #         print('set all param.grad = None but skip optimizer.step()')
        #     for param in model.parameters():
        #         if param.requires_grad:
        #             param.grad = None
        if i in range(3000,4000):
            if args.rank == 1 and i == 3001:
                print('skip optimizer.step() from now on')
            pass

        if i % 20 == 0:
            torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and (i % args.print_freq == 0 or i == max_iter - 1):
            progress.print(i)



class AverageMeter(object):
    """Computes and stores the average and current value
       When length <=0 , save all history data """

    def __init__(self, name, fmt=':f', length=0):
        self.name = name
        self.fmt = fmt
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 1:
            self.history = []
        elif self.length <= 0:
            self.count = 0
            self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val):
        self.val = val
        if self.length > 1:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.avg = np.mean(self.history)
        elif self.length <= 0:
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

    def __str__(self):
        if self.length == 1:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), need_raw=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_raw = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res_raw.append(correct_k)
            res.append(correct_k.mul(100.0 / batch_size))
        if need_raw:
            res.extend(res_raw)
        return res


if __name__ == '__main__':
    main()
