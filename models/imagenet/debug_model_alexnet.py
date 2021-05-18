import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models
import numpy as np
from termcolor import colored



code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')

def get_data(data, reduction='mean'):
    assert reduction in ['mean', 'sum', 'all']
    if isinstance(data, (torch.Tensor)):
        data = data.clone().float()
        if reduction == 'mean':
            return data.detach().mean().cpu()
        elif reduction == 'sum':
            return data.detach().abs().mean().cpu()
        else:
            return data.detach().abs().cpu()
    elif isinstance(data, dict):
        return {k: get_data(v, reduction) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        return data.__class__(get_data(v, reduction) for v in data)
    else:
        return data

def get_data2(data, reduction='mean'):
    assert reduction in ['mean', 'sum', 'all']
    if isinstance(data, (torch.Tensor)):
        npdata = data.clone().detach().cpu().numpy()
        if reduction == 'mean':
            return np.mean(npdata)
        elif reduction == 'sum':
            return np.sum(npdata)
        else:
            return npdata.flatten()[:10]
    elif isinstance(data, dict):
        return {k: get_data(v, reduction) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        return data.__class__(get_data(v, reduction) for v in data)
    else:
        return data

def hook(name, mm, tag='forward'):
    hook_func = get_data2
    def inner_hook(m, input, output):
        print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(input), hook_func(output)))
        #  print("Layer {} {}:\n input {}\n output {}\n".format(name, mm, tag, get_data2(input), get_data2(output)))
    return inner_hook

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
            # nn.ReLU(inplace=False),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
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
            # #x = nn.functional.adaptive_avg_pool2d(x, (6,6))
            x = torch.flatten(x, 1)
            x = x.cuda()
        else:
            x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__=="__main__":
    
    # m = models.vgg_min.vgg16()
    # m = models.alexnet()
    m = AlexNet()

    input = torch.randn(2, 3, 224, 224, requires_grad=True)
    
    #进行hook注册访问每一层的forward和backward输入输出
    for name, mm in m.named_modules():
        print(code_blue(name), "---", code_blue(mm))
        mm.register_forward_hook(hook(name, mm))
        mm.register_backward_hook(hook(name, mm, tag='backward'))

   # pytorch环境下保存模型参数和输入
    if torch.__version__ != "parrots":
        torch.save(m.state_dict(), 'data/checkpoint.pth')
        torch.save(input, 'data/input.pth')
    # parrots环境下固定模型参数和输入 
    else:
        input = torch.load('data/input.pth')
        m.load_state_dict(torch.load('data/checkpoint.pth'))
    m = m.train()
    if torch.__version__ == "parrots":
    #    torch.set_printoptions(6)
       m = m.to_memory_format(torch.channels_last)
       input = input.contiguous(torch.channels_last)
       input = input.cuda()
       m = m.cuda()
    elif torch.__version__ == "1.3.0a0":
        torch.set_printoptions(10)
        import torch_mlu.core.mlu_model as ct
        ct.set_cnml_enabled(False)
        m = m.to(ct.mlu_device())
        input = input.to(ct.mlu_device())
    elif torch.__version__ == "1.3.1": # pytorch cpu
        pass

    out = m(input)
    out.backward(torch.ones_like(out))
    print(out.abs().sum().cpu())