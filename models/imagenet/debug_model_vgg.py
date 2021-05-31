import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import models.vgg_min
import numpy as np
from hook_function import hookCompareTool, code_blue

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()

        self.conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu = nn.ReLU(False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu2 = nn.ReLU(False)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(False),
            # nn.Dropout(p),
            nn.Linear(4096, 4096),
            nn.ReLU(False),
            # # nn.Dropout(p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool(x)
        x = x.cpu()
        
        x = x.view(x.size(0), -1)
        
        if torch.__version__ == "parrots":
            x = x.cuda()
        elif torch.__version__ == "1.3.0a0":
            import torch_mlu.core.mlu_model as ct
            x = x.to(ct.mlu_device())
        elif torch.__version__ == "1.3.1":
            pass
        x = self.classifier(x)
        return x


if __name__=="__main__":
    
    hct = hookCompareTool()
    
    # m = models.vgg_min.vgg16()
    # m = models.vgg16(p=0)
    m = VGG()

    # input = torch.randn(2, 3, 224, 224, requires_grad=True)
    input = torch.randn(2, 512, 14, 14, requires_grad=True)
    # input = torch.randn(2, 512 * 7 * 7, requires_grad=True)
    # input = torch.randn(2, 512, 7, 7, requires_grad=True)
    
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