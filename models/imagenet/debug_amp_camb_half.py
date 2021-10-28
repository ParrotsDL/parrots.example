import math
import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import numpy as np
import copy
import argparse
import yaml
from addict import Dict
import torch.distributed as dist
from utils.dataloader import build_dataloader
import torch.cuda.amp as amp
from parrots.base import use_camb, use_cuda

torch_version = torch.__version__

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--pth_path', default='',
                    type=str, help='path to checkpoint file')
parser.add_argument('--config', default='configs/resnet50.yaml',
                    type=str, help='path to config file')
parser.add_argument('--use_amp', dest='use_amp', action='store_true',
                    help='use amp for auto mixed percision')

args = parser.parse_args()
args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
cfgs = Dict(args.config)

args.pth_path = "/share1/fengsibo/parrots.example/models/imagenet/checkpoints/resnet50_1019/resnet50_best.pth"
cfgs.dataset.batch_size = 2

train_loader, test_loader = build_dataloader(cfgs.dataset, 1, "MemcachedReader")
input, target  = next(iter(train_loader))

def load_checkpoint(model):
    if len(args.pth_path) > 0:
        state_dict = torch.load(args.pth_path, map_location='cpu')
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict['state_dict'])

scaler = amp.GradScaler(2**17)

if __name__== "__main__":

    SEED = 42
    torch.manual_seed(SEED)

    model = models.resnet50()
    load_checkpoint(model)

    state_dict = model.state_dict()
    for m in state_dict:
        print(m, state_dict[m].sum())

    model = model.cuda()
    input = torch.ones(2, 3, 224, 224, requires_grad=True)
    input = input.cuda()
    if use_cuda:
        target = torch.ones_like(target)
        target = target.cuda()
    else:
        target = torch.ones_like(target)
        target = target.int().cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    if use_camb:
        input = input.contiguous(torch.channels_last)
        model = model.to_memory_format(torch.channels_last)

    with amp.autocast(enabled=True):
        output = model(input)

    loss = criterion(output.float(), target)

    print(f"output: {output}, output_sum: {torch.sum(output).item()}, target: {target}, target_sum: {torch.sum(target)}")

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
