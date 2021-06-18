import torch as torch
import torch.nn as nn
from torch.nn import init
import pdb

__all__ = ["shuffle_v2"]


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1,
            bias=True,
            groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=bias,
        groups=groups)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2)
    x = x.contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def channel_split(x, splits=[24, 24]):
    return torch.split(x, splits, dim=1)


class ParimaryModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=24):
        super(ParimaryModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ParimaryModule = nn.Sequential(
            conv3x3(in_channels, out_channels, 2, 1, True, 1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ParimaryModule(x)
        return x


class FinalModule(nn.Module):
    def __init__(self, in_channels=464, out_channels=1024, num_classes=1000):
        super(FinalModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(out_channels, num_classes)
        self.FinalConv = nn.Sequential(
            conv1x1(in_channels, out_channels, True, 1),
            nn.BatchNorm2d(out_channels), nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.FinalConv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, splits_left=2):
        super(ShuffleNetV2Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.splits_left = splits_left

        if stride == 2:
            self.Left = nn.Sequential(
                conv3x3(in_channels, in_channels, stride, 1, True,
                        in_channels), nn.BatchNorm2d(in_channels),
                conv1x1(in_channels, out_channels // 2, True, 1),
                nn.BatchNorm2d(out_channels // 2), nn.ReLU())
            self.Right = nn.Sequential(
                conv1x1(in_channels, in_channels, True, 1),
                nn.BatchNorm2d(in_channels), nn.ReLU(),
                conv3x3(in_channels, in_channels, stride, 1, True,
                        in_channels), nn.BatchNorm2d(in_channels),
                conv1x1(in_channels, out_channels // 2, True, 1),
                nn.BatchNorm2d(out_channels // 2), nn.ReLU())
        elif stride == 1:
            in_channels = in_channels - in_channels // splits_left
            self.Right = nn.Sequential(
                conv1x1(in_channels, in_channels, True, 1),
                nn.BatchNorm2d(in_channels), nn.ReLU(),
                conv3x3(in_channels, in_channels, stride, 1, True,
                        in_channels), nn.BatchNorm2d(in_channels),
                conv1x1(in_channels, in_channels, True, 1),
                nn.BatchNorm2d(in_channels), nn.ReLU())
        else:
            raise ValueError('stride must be 1 or 2')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if self.stride == 2:
            x_left, x_right = x, x
            x_left = self.Left(x_left)
            x_right = self.Right(x_right)
        elif self.stride == 1:
            x_split = channel_split(x, [
                self.in_channels // self.splits_left,
                self.in_channels - self.in_channels // self.splits_left
            ])
            x_left, x_right = x_split[0], x_split[1]
            # x_right = self.Right(x_right)

        print(x_left.shape, x_right.shape)
        x = torch.cat((x_left, x_right), dim=1)
        # x = channel_shuffle(x, 2)
        if torch.__version__ == "1.6.0a0":
            x = channel_shuffle(x, 2)
        else:
            x = torch.shuffleChannel(x, 2)
        return x


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=1000,
                 net_scale=1.0,
                 stage_repeat=1,
                 splits_left=2):
        super(ShuffleNetV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net_scale = net_scale
        self.splits_left = splits_left

        self.out_channels = [24, 116, 232, 464, 1024]
    

        self.ParimaryModule = ParimaryModule(in_channels, self.out_channels[0])

        self.Stage1 = self.Stage(1, [1, 3])
        self.Stage2 = self.Stage(2, [1, 7])
        self.Stage3 = self.Stage(3, [1, 3])

        self.FinalModule = FinalModule(self.out_channels[3],
                                       self.out_channels[4], num_classes)

    def Stage(self, stage=1, BlockRepeat=[1, 3]):
        modules = []

        if BlockRepeat[0] == 1:
            modules.append(
                ShuffleNetV2Block(self.out_channels[stage - 1],
                                  self.out_channels[stage], 
                                  stride=2,
                                  splits_left=self.splits_left))
        else:
            raise ValueError('stage first block must only repeat 1 time')

        for i in range(BlockRepeat[1]):
            modules.append(
                ShuffleNetV2Block(self.out_channels[stage],
                                  self.out_channels[stage], 1,
                                  self.splits_left))

        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.ParimaryModule(x)
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.FinalModule(x)
        return x


def shuffle_v2(**kwargs):
    model = ShuffleNetV2(**kwargs)
    return model
