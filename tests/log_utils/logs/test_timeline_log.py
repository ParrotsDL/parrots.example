import argparse
import torch
import parrots
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='test for timeline')
parser.add_argument("--export", type=bool, default=False)
args = parser.parse_args()


if args.export is False:
    parrots.runtime.profile_function(enable=False)
    parrots.runtime.profile_attrs(enable=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


parrots.runtime.profile(enable=True, file='profile.txt')

with parrots.record_time('test_timeline'):
    net = Net().cuda()
    input = torch.randn(1, 1, 32, 32).cuda()
    input.requires_grad = True
    out = net(input)
    out.backward()

parrots.runtime.profile(enable=False)