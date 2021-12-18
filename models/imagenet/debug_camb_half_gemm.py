import math
import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import numpy as np
import models.imagenet.hook as hook
from pape.half.half_model import HalfModel
import copy

torch_version = torch.__version__

hook.CAMB_TORCH_VERSION = "1.6.0a0+ab70945" # no use
hook.CPU_PYTORCH_VERSION = "1.3.1+cpu"


class AlexNet(nn.Module):

    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=isBias),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=isBias),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=isBias),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256, 4096, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=False),
        )

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model


if __name__== "__main__":

    hc = hook.hookCompare()

    # m = models.alexnet()
    # m = models.resnet50()
    # m = models.resnet18()
    # m = models.vgg16()
    m = alexnet()
    input = torch.randn(2, 256, requires_grad=True)


    # m = models.inception_v3()
    # m = inception_v3()
    # input = torch.randn(2, 3, 299, 299, requires_grad=True)

    source_model = copy.deepcopy(m)
    
    for name, mm in m.named_modules():
        print(hook.code_blue(name), "---", hook.code_blue(mm))
        mm.register_forward_hook(hc.hook(name, mm))
        # mm.register_backward_hook(hc.hook(name, mm, tag='backward'))

    input, m = hc.save_and_load(input, m)
    m.train()
    input, m = hc.to_cuda(input, m, qb=16)

    # def pthInfo(m):
    #     state_dict = m.state_dict()
    #     for p in state_dict:
    #         print(p, state_dict[p].shape, state_dict[p].dtype)

    
    if torch_version == hook.CAMB_PARROTS_VERSION: # parrots
        m = HalfModel(m)
        input = input.half()
        pass
    elif torch_version == hook.CPU_PYTORCH_VERSION: # cpu pytorch
        input = input.half()
        m = m.half()
    else: # camb pytorch
        m = HalfModel(m)
        # pthInfo(m)
        input = input.half()
        pass
    
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
    out = m(input)

    # scale = 65500.0
    scale = 1.0
    loss = torch.ones_like(out) * scale
    out.backward(loss)
    # loss.backward()
    optimizer.step()
    
    hc.save_and_compare_hook()

    # hc.save_updated_model(m)
    # hc.compare_updated_model(source_model)

    print("Inference result:", hook.code_green(out.float().abs().sum().cpu()))