import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import numpy as np
from hook_function import hookCompareTool, code_blue

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=False),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if torch.__version__ == "parrots":
            x = x.cpu()#.contiguous()
            x = torch.flatten(x, 1)
            x = x.cuda()
        else:
            x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__=="__main__":

    hct = hookCompareTool()
    
    # m = models.vgg_min.vgg16()
    # m = models.alexnet()
    m = AlexNet()

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