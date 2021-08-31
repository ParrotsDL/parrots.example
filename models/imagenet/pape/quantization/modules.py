from torch.nn import Module
import torch


class Cat(Module):
    def __init__(self, dim=0):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, seq):
        return torch.cat(seq, dim=self.dim)


class EltwiseAdd(Module):
    def __init__(self, inplace=False):
        super(EltwiseAdd, self).__init__()

        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class EltwiseMult(Module):
    def __init__(self, inplace=False):
        super(EltwiseMult, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res
