import torch
import torch.nn.functional as F
import torch.nn as nn
from pape.half.half_model import HalfModel
# from pape.half.half_optimizer import HalfOptimizer
import models
import time
import numpy as np
from termcolor import colored

code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')

def get_data_numpy(data, reduction='mean'):
        if isinstance(data, (torch.Tensor)):
            npdata = data.clone().detach().cpu().numpy()
            if reduction == 'mean':
                return np.mean(npdata)
            elif reduction == 'sum':
                return np.sum(npdata)
            elif reduction == 'shape':
                return npdata.shape
            else:
                return npdata.reshape(-1)[-5:], np.mean(npdata), npdata.shape
        elif isinstance(data, dict):
            return {k: get_data_numpy(v) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return data.__class__(get_data_numpy(v) for v in data)
        else:
            return data

def hook(name, mm, tag='forward'):
    hook_func = get_data_numpy
    def inner_hook(m, input, output):
        if tag == 'forward':
            print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(input), hook_func(output)))
        else:
            print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(output), hook_func(input)))
    return inner_hook


class testModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(testModel, self).__init__()

        self.conv = nn.Conv2d(3, 3, 3, bias=False)
        self.bn = nn.BatchNorm2d(3, eps=0.001)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


if __name__=="__main__":

    USE_HALF = True
    iters = 1000

    # torch.cuda.syncronize()
    
    # m = testModel()
    m = models.resnet50()

    input = torch.randn(2, 3, 224, 224, requires_grad=True)

    # if USE_HALF:
    #     input = input.half()
    #     m = HalfModel(m)
    #     # m = m.half()

    # print(input)
    # for param in m.parameters():
    #     print(param)

    # for name, mm in m.named_modules():
    #     mm.register_forward_hook(hook(name, mm))
    #     mm.register_backward_hook(hook(name, mm, tag='backward'))

    m = m.to_memory_format(torch.channels_last)
    m = m.cuda()

    input = input.contiguous(torch.channels_last)
    input = input.cuda()

    start_time = time.time()
    
    for i in range(iters):
        out = m(input)
        print(i)
        # out.backward(torch.ones_like(out))

    torch.cuda.synchronize()
    end_time = time.time()

    print("cost time: {:.4f}".format((end_time - start_time) / iters))

    # print(out)