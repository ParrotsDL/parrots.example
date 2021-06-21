import torch
import models
import torch.nn.functional as F
import torch.nn as nn
# import models
import models.mobile_v2_min
import numpy as np
from hook_function import hookCompareTool, code_blue


class mv2(nn.Module):
    def __init__(self, num_classes=1000):
        super(mv2, self).__init__()

        # self.conv = nn.Conv2d(96, 96, 3, 1, 1, groups=1)
        self.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=True)
        # self.batch_norm = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        # x = self.batch_norm(x)
        x = self.relu(x)
        return x

if __name__=="__main__":
    
    hct = hookCompareTool()

    m = models.mobile_v2()
    # m = models.mobile_v2_min.mobile_v2()
    # m = mv2()

    input = torch.randn(2, 3, 224, 224, requires_grad=True)

    # 进行hook注册访问每一层的forward和backward输入输出
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
