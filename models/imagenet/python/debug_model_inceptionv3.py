import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models.inception_v3_min
import numpy as np
import pdb
from hook_function import hookCompareTool, code_blue

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        x = torch.cat(outputs, 1)
        return x

class incv3(nn.Module):
    def __init__(self, num_classes=1000):
        super(incv3, self).__init__()

        self.Mixed_5b = InceptionA(288, pool_features=64)
        self.Mixed_5c = InceptionA(288, pool_features=64)


    def forward(self, x):

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        
        return x


if __name__=="__main__":

    hct = hookCompareTool()
    
    m = models.inception_v3()
    # m = models.inception_v4()
    # m = models.densenet121()
    # m = models.inception_v3_min.inception_v3()
    # m = incv3()

    # input = torch.randn(2, 288, 35, 35, requires_grad=True)
    input = torch.randn(2, 3, 299, 299, requires_grad=True)
    

    # 进行hook注册访问每一层的forward和backward输入输出
    for name, mm in m.named_modules():
        print(code_blue(name), "---", code_blue(mm))
        mm.register_forward_hook(hct.hook(name, mm))
        mm.register_backward_hook(hct.hook(name, mm, tag='backward'))

    input, m = hct.save_and_load(input, m)
    m = m.train()
    input, m = hct.to_cuda(input, m)
    out = m(input)
    out.backward(torch.ones_like(out))

    hct.save_and_compare_hook()

    print(out.abs().sum().cpu())