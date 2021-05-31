import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import models.senet_min
import numpy as np
from hook_function import hookCompareTool, code_blue

if __name__=="__main__":

    hct = hookCompareTool()
    
    m = models.senet_min.se_resnet50()

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

    hct.save_and_compare_hook()

    print(out.abs().sum().cpu())