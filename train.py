# -*- coding: utf-8 -*-

'''
Janurary 2018 by Wei Li
liweihfyz@sjtu.edu.cn
https://www.github.cim/leviswind/transformer-pytorch
'''
from __future__ import print_function

import argparse
import random
import os
import time
import re
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler


from AttModel import AttModel
from data_load import TrainDataSet, load_vocab
from hyperparams import Hyperparams as hp
from util import *

try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    import torch_mlu.core.mlu_quantize as qt
except ImportError:
    print("import torch_mlu failed!")

if torch.__version__ == "parrots":
    from parrots.base import use_camb
else:
    use_camb = False
int_dtype = torch.int if use_camb else torch.long

logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("__name__")

def main(args):
    if args.launcher == 'slurm':
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        node_parts = re.findall('[0-9]+', node_list)
        os.environ['MASTER_ADDR'] = f'{node_parts[0]}.{node_parts[1]}.{node_parts[2]}.{node_parts[3]}'
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

    if not os.path.exists(args.ckp_path):
        os.makedirs(args.ckp_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if args.seed is not None:
        set_seed(args.seed)

    import time
    start = time.time()
    if args.device == "mlu":
        ct.set_device(0 if args.rank == -1 else args.rank)
        ct.set_cnml_enabled(False)
    if args.device == "gpu":
        torch.cuda.set_device(0 if args.rank == -1 else args.rank)
    # distributed training env setting up
    if args.dist:
        dist.init_process_group(backend='cncl' if use_camb or args.device == "mlu" else 'nccl', rank=args.rank, world_size=args.world_size)

    startepoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        startepoch = state['epoch'] + 1

    # Load data
    source_train = args.dataset_path + hp.source_train
    target_train = args.dataset_path + hp.target_train
    train_dataset =  TrainDataSet(source_train, target_train, args.vocab_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)#, num_replicas = args.world_size, rank = args.rank)
    # args.batch_size = args.batch_size // args.world_size

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size= args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler)

    hp.dropout_rate = args.dropout_rate

    # distributed model
    de2idx, idx2de = load_vocab(os.path.join(args.vocab_path, "de.vocab.tsv"))
    en2idx, idx2en = load_vocab(os.path.join(args.vocab_path, "en.vocab.tsv"))
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)
    model = AttModel(hp, enc_voc, dec_voc)

    # adaptive_quantize
    if args.device == "mlu" and not getattr(args, 'max_bitwidth', False):
        model = qt.adaptive_quantize(model, len(train_loader), bitwidth=args.bitwidth)
    if args.device == "mlu":
        model.to(ct.mlu_device())
    if args.device == "gpu":
        if args.quantify:
            from torch.utils import quantize
            model = quantize.convert_to_adaptive_quantize(
                model, len(train_loader))
        model.cuda()

    # load state_dict
    if args.resume:
        model.load_state_dict(state['model'], strict=False)

    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0 if args.rank == -1 else args.rank])

    logger.info(model)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr, betas=[0.9, 0.98], eps=1e-8)
    if args.resume:
        optimizer.load_state_dict(state['optim'])

        if args.device == "mlu":
            ct.to(optimizer, torch.device('mlu'))

    if args.device == "gpu":
        cudnn.benckmark = True

    if args.seed is not None:
        set_seed(args.seed)

    for epoch in range(startepoch, args.num_epochs + 1):
        if args.dist:
            train_sampler.set_epoch(epoch)
        epoch_log = os.path.join(args.log_path, "epoch{:02d}_rank{:02d}.txt".format(epoch, -1))

        epoch_iters = len(train_loader)

        train(train_loader, model, optimizer, epoch, args, epoch_log, args.rank, epoch_iters)

        # save model
        if not args.save_ckpt:
            break
        if not args.dist or ( args.dist and args.rank == 0 ):
            checkpoint_path = os.path.join(args.ckp_path, "model_epoch_{:02d}.pth".format(epoch))
            state = {}
            state['epoch'] = epoch
            if args.dist:
                state['model'] = model.module.state_dict()
            else:
                state['model'] = model.state_dict()
            state['optim'] = optimizer.state_dict()
            torch.save(state, checkpoint_path)

    end = time.time()
    print("Using Time: " + str(end-start))

def set_lr(optimizer, args, cur_iters):
    if cur_iters <= args.warmup_iters:
        lr = float(args.lr * cur_iters) / float(args.warmup_iters)
    else:
        lr = float(args.lr * args.warmup_iters**0.5 * cur_iters ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, optimizer, epoch, args, epoch_log, rank, epoch_iters):
    adaptive_cnt = int(os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT')) if (
            os.getenv('MLU_ADAPTIVE_STRATEGY_COUNT') is not None) else 0
    batch_time_benchmark = []
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')
    acces = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acces],
        prefix="Card{} Epoch: [{}]".format(rank, epoch))

    model.train()

    if args.device == "gpu":
        torch.cuda.synchronize()
    end = time.time()

    cur_iters = (epoch - 1) * epoch_iters + 1
    for i, (data, target) in enumerate(train_loader):
        set_lr(optimizer, args, cur_iters)
        if (i == args.iterations) and rank == 0:
            logger.info('The program iteration runs out. iterations: %d' % args.iterations)
            break

        data_time.update(time.time() - end)
        if args.device == "gpu":
            data = data.to(dtype=int_dtype).cuda()
            target = target.to(dtype=int_dtype).cuda()
        if args.device == "mlu":
            data = data.to(ct.mlu_device(), non_blocking=True)
            target = target.to(ct.mlu_device(), non_blocking=True)

        loss, _, acc = model(data, target)
        losses.update(loss.item())
        acces.update(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.device == "gpu":
            torch.cuda.synchronize()
        if i >= adaptive_cnt and adaptive_cnt > 0:
            batch_time_benchmark.append(time.time() - end)
        batch_time.update(time.time() - end)
        end = time.time()

        if rank <= 0:
           msglog(epoch_log, "{}, {}".format(loss.item(), acc.item()))
        if i % args.print_freq == 0 and rank == 0:
            progress.display(i)
    if args.device == "mlu":
        cards = ct.device_count() if rank == 0 else 1
    if args.device == "gpu":
        cards = torch.cuda.device_count() if rank == 0 else 1
    if ((args.distributed == False) or (rank == 0)) and os.getenv('AVG_LOG'):
        with open(os.getenv('AVG_LOG'), 'a') as train_avg:
            train_avg.write('net:transformer, iter:{}, cards:{}, avg_loss:{}, avg_time:{}, '.format(args.iterations,
                            cards, losses.avg, batch_time.avg))
    if ((args.distributed == False) or (rank == 0)) and os.getenv('BENCHMARK_LOG') and args.device == "mlu":
        with open(os.getenv('BENCHMARK_LOG'), 'a') as train_avg:
            line = 'network:transformer, Batch Size:{}, device count:{}, Precision:{}, DPF mode:{}, \
                time_avg:{:.3f}s, time_var:{:.6f}, throughput(fps):{:.1f}'
            line_after = re.sub(' +', ' ', line)
            train_avg.write(line_after.format(args.batch_size, cards,
                "O0",
                "ddp" if args.distributed == True else "single",
                np.mean(batch_time_benchmark),
                np.var(batch_time_benchmark),
                args.batch_size * hp.maxlen / np.mean(batch_time_benchmark) * cards) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer training.")
    parser.add_argument('--seed', default=66, type=int, help='random seed.')
    parser.add_argument('--log-path', default='logs', type=str, help='training log path.')
    parser.add_argument('--ckp-path', default='models', type=str, help='training ckps path.')
    parser.add_argument('--resume', type=str, help='resume ckp path.')
    parser.add_argument('--batch-size', default=32, type=int, help='training batch size for all')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers.')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training.')
    parser.add_argument('--rank', default=0, type=int, help='node rank fro distributed training.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency of information.')
    parser.add_argument('--distributed', action='store_true', help='distributed training.')
    parser.add_argument('--save_ckpt', default=True, type=bool, help='save checkpoint.')
    parser.add_argument('--device', default='gpu', type=str, help='set the type of hardware used for training.')
    parser.add_argument('--bitwidth', default=8, type=int, help="Set the initial quantization width of network training.")
    parser.add_argument('--iterations', default=-1, type=int, help="Number of training iterations.")
    parser.add_argument('--dataset-path', default='/mnt/lustre/share/nlp/corpora/', type=str, help='The path of imagenet dataset.')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of training num_epochs.')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate.')
    parser.add_argument('--master-addr', default='127.0.0.1', type=str, help='ddp address.')
    parser.add_argument('--master-port', default='29500', type=str, help='ddp address port.')
    parser.add_argument('--max_bitwidth', action='store_true', help='use Max Bitwidth of MLU training')
    parser.add_argument('--warmup-iters', default=300, type=float, help='warm up iterations')
    parser.add_argument('--lr', "--learning-rate", default=0.0005, type=float, help="learning rate for training")
    parser.add_argument('--launcher', type=str, default="slurm", choices=['slurm', 'mpi'], help='distributed backend')
    parser.add_argument('--port', default=12345, type=int, metavar='P', help='master port')
    parser.add_argument('--quantify', dest='quantify', action='store_true', help='quantify training')
    parser.add_argument('--vocab_path', type=str, default="./data/IWSLT/preprocessed", help='the path to preprocessed data')
    args = parser.parse_args()
    main(args)

