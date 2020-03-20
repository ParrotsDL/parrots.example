import torch.nn as nn
import torch.nn.functional as F
from pape.op import SplitBatchNorm2d
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, num_groups=1, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = SplitBatchNorm2d(planes, num_groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = SplitBatchNorm2d(planes, num_groups)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                SplitBatchNorm2d(self.expansion*planes, num_groups)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, num_groups=1, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = SplitBatchNorm2d(planes, num_groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = SplitBatchNorm2d(planes, num_groups)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = SplitBatchNorm2d(self.expansion*planes, num_groups)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                SplitBatchNorm2d(self.expansion*planes, num_groups)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_groups=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SplitBatchNorm2d(64, num_groups)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], num_groups,  stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], num_groups, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], num_groups, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], num_groups, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SplitBatchNorm2d):
                for bn in m.bn_list:
                    bn.weight.data.fill_(1)
                    bn.bias.data.zero_()
                    # m.bns[g].cuda()

    def _make_layer(self, block, planes, num_blocks, num_groups, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, num_groups, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_splitbn(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
