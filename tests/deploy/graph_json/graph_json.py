import torch
import parrots
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.sigmoid(x)
        x = self.relu(x)
        y = y + 1
        x += y
        return x


def dump_json(net):
    net.eval()
    input = torch.randn(1, 3, 224, 224)
    parrots_graph = parrots.tracking(net, input)
    parrots.dump_graph(parrots_graph, './test')


if __name__ == '__main__':
    net = Net()
    dump_json(net)
