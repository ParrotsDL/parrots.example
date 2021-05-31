import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import models.shuffle_v2_min
import numpy as np
from hook_function import hookCompareTool, code_red, code_blue

class Test(nn.Module):
    def __init__(self, num_classes=1000):
        super(Test, self).__init__()

    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        return x


if __name__=="__main__":

    hct = hookCompareTool()
    
    # m = models.shuffle_v1()
    m = models.shuffle_v2_min.shuffle_v2()
    m = Test()

    input = torch.randn(2, 3, 224, 224, requires_grad=True)
    
    #进行hook注册访问每一层的forward和backward输入输出
    for name, mm in m.named_modules():
        print(code_blue(name), "---", code_blue(mm))
        mm.register_forward_hook(hct.hook(name, mm))
        mm.register_backward_hook(hct.hook(name, mm, tag='backward'))

    input, m = hct.save_and_load(input, m)
    m = m.train()
    input, m = hct.to_cuda(input, m)
    out = m(input)
    out.backward(torch.ones_like(out))

    # hct.save_and_compare_hook()

    # print(out.abs().sum().cpu())