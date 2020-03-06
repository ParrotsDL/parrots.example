import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class OneHotNLLLoss(_Loss):
    def __init__(self, num_classes):
        super(OneHotNLLLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1)

        loss = - torch.sum(F.log_softmax(input, 1) * (one_hot.detach())) / input.size(0)
        return loss
