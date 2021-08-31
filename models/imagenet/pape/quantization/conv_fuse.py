import torch.nn as nn


class Conv_BN(nn.Module):
    def __init__(self, conv, bn, merge_bn=False):
        super(Conv_BN, self).__init__()
        self.conv = conv
        self.bn = bn
        self.merge_bn = merge_bn

    def forward(self, x):
        if not self.merge_bn:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, conv, bn, relu, merge_bn=False):
        super(ConvBNReLU, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.merge_bn = merge_bn

    def forward(self, x):
        if not self.merge_bn:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        out = self.relu(out)
        return out
