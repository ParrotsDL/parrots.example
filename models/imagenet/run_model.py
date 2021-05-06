import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import models.vgg_min
import numpy as np

class DPN(nn.Module):
    def __init__(self, num_classes=1000):
        super(DPN, self).__init__()

        # self.conv = nn.Conv2d(96, 96, 3, 1, 1, groups=1)
        self.conv = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
            groups=96, bias=False)

    def forward(self, x):
        x = self.conv(x)
        
        print(x.shape)
        return x

if __name__=="__main__":
    
    m = DPN()

    # input = torch.randn(2, 512, 7, 7, requires_grad=True)
    input = torch.ones(2, 96, 56, 56, requires_grad=True)
    
    m = m.train()
    
    m = m.to_memory_format(torch.channels_last)
    input = input.contiguous(torch.channels_last)
    input = input.cuda()
    m = m.cuda()
    
    out = m(input)