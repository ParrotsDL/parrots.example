import torch
import torch.nn.functional as F
import torch.nn as nn
from pape.half.half_model import HalfModel
# from pape.half.half_optimizer import HalfOptimizer
import models
import time
import numpy as np
from termcolor import colored
import math

code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')

import torch.cuda.amp as amp
scaler = amp.GradScaler()

if __name__=="__main__":

    m = models.alexnet()
    m = m.to_memory_format(torch.channels_last)
    m = m.cuda()

    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

    input = torch.randn(2, 3, 224, 224, requires_grad=True)
    input = input.contiguous(torch.channels_last)
    input = input.cuda()

    out_fp32 = m(input)
    
    with amp.autocast():
        out = m(input)
        
    loss = torch.ones_like(out) * 1.0 - out
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # out.backward(loss)
    # optimizer.step()