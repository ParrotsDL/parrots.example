import math
import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import numpy as np
import models.imagenet.hook as hook
import copy
import argparse
import yaml
from addict import Dict
import torch.distributed as dist

from pape.half.half_model import HalfModel
from utils.dataloader import build_dataloader

torch_version = torch.__version__

hook.CAMB_TORCH_VERSION = "1.6.0a0+ab70945" # no use
hook.CPU_PYTORCH_VERSION = "1.3.1+cpu"

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--pth_path', default='',
                    type=str, help='path to checkpoint file')
parser.add_argument('--config', default='configs/resnet50.yaml',
                    type=str, help='path to config file')
args = parser.parse_args()
args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
cfgs = Dict(args.config)

cfgs.dataset.batch_size = 2

train_loader, test_loader = build_dataloader(cfgs.dataset, 1, "MemcachedReader")
input, target  = next(iter(train_loader))

def load_checkpoint(model):
    if len(args.pth_path) > 0:
        state_dict = torch.load(args.pth_path, map_location='cpu')
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict['state_dict'])

isBias = True
isBias = False
class AlexNet(nn.Module):

    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=isBias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=isBias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=isBias),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     # nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096, bias=False),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(),
        #     nn.Linear(4096, 4096, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes, bias=False),
        # )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model

if __name__== "__main__":

    hc = hook.hookCompare()

    model = alexnet()
    input = torch.randn(2, 3, 224, 224, requires_grad=True)

    load_checkpoint(model)
    criterion = nn.CrossEntropyLoss()
    source_model = copy.deepcopy(model)
    
    # 注册 hook
    for name, mm in model.named_modules():
        print(hook.code_yellow(name), "---", hook.code_green(mm))
        mm.register_forward_hook(hc.hook(name, mm))
        mm.register_backward_hook(hc.hook(name, mm, tag='backward'))

    input, target, model = hc.save_and_load(input, target, model)
    model.train()

    if torch_version == hook.CAMB_PARROTS_VERSION: # parrots
        model = model.to_memory_format(torch.channels_last)
        model = model.cuda()
        load_checkpoint(model)
        input = input.cuda()
        if input.ndims == 4:
            input = input.contiguous(torch.channels_last)
        target = target.int().cuda()
        criterion = criterion.cuda()
        HalfModel(model)
        input = input.half()
        pass
    elif torch_version == hook.CPU_PYTORCH_VERSION: # cpu pytorch
        # input = input.half()
        # model = model.half()
        pass
    else: # camb pytorch
        torch.set_printoptions(10)
        import torch_mlu.core.mlu_model as ct
        # ct.set_device(4)
        ct.set_cnml_enabled(False)
        ct.set_quantized_bitwidth(16)
        model = model.to(ct.mlu_device())
        input = input.to(ct.mlu_device())
        target = target.to(ct.mlu_device())
        criterion = criterion.to(ct.mlu_device())
        HalfModel(model)
        input = input.half()
        pass

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    output = model(input)
    output = output.float()

    scale = 2**16
    # scale = 1.0

    # loss = criterion(output, target) * scale
    # loss = loss.half()
    # optimizer.zero_grad()
    # loss.backward()
    
    # loss = torch.ones_like(output)
    output.backward(torch.ones_like(output) * scale)
    
    # optimizer.step()
    
    hc.save_and_compare_hook()
    # hc.save_updated_model(model)
    # hc.compare_updated_model(source_model)

    print("Inference result:", hook.code_green(output.float().abs().sum().cpu()))