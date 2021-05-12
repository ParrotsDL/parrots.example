import torch
import models
import torch.nn.functional as F
import torch.nn as nn
# import models
import models.mobile_v2_min
import numpy as np
from termcolor import colored
from collections import OrderedDict

code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')

print("pytorch version:", code_red(torch.__version__))

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
        data = data.clone().detach().cpu().numpy()
        if reduction == 'mean':
            return np.mean(data)
        elif reduction == 'sum':
            return np.sum(data)
        else:
            return data
    elif isinstance(data, dict):
        return {k: get_data(v, reduction) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        return data.__class__(get_data(v, reduction) for v in data)
    else:
        return data

def hook(name, mm, tag='forward'):
    hook_func = get_data2
    def inner_hook(m, input, output):
        if tag == 'forward':
            print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(input), hook_func(output)))
        else:
            print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(output), hook_func(input)))
    return inner_hook

class mv2(nn.Module):
    def __init__(self, num_classes=1000):
        super(mv2, self).__init__()

        # self.conv = nn.Conv2d(96, 96, 3, 1, 1, groups=1)
        # self.conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False)
        self.conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=True)
        self.batch_norm = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(False)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

if __name__=="__main__":
    
    # m = models.mobile_v2()
    # m = models.mobile_v2_min.mobile_v2()
    m = mv2()

    # input = torch.randn(2, 3, 224, 224, requires_grad=True)
    input = torch.randn(2, 32, 224, 224, requires_grad=True)
    
    #进行hook注册访问每一层的forward和backward输入输出
    for name, mm in m.named_modules():
        print(code_blue(name), "---", code_blue(mm))
        mm.register_forward_hook(hook(name, mm))
        mm.register_backward_hook(hook(name, mm, tag='backward'))

    # pytorch环境下保存模型参数和输入
    if torch.__version__ == "1.3.0a0":
        torch.save(m.state_dict(), 'data/checkpoint.pth')
        torch.save(input, 'data/input.pth')
    # parrots环境下固定模型参数和输入 
    else:
        input = torch.load('data/input.pth')
        m.load_state_dict(torch.load('data/checkpoint.pth'))

    # # pytroch 重加载上次的权重
    # input = torch.load('data/input.pth')

    # load_dict = torch.load('data/checkpoint.pth')
    # model_dict = m.state_dict()
    # new_state_dict = OrderedDict()

    # for i, (k, v) in enumerate(load_dict.items()):
    #     if k in model_dict:
    #         print("======", k)
    #         new_state_dict[k] = v
    # m.load_state_dict(new_state_dict)


    m = m.train()
    if torch.__version__ == "parrots":
    #    torch.set_printoptions(6)
       m = m.to_memory_format(torch.channels_last)
       input = input.contiguous(torch.channels_last)
    #    input = input.transpose(0, 2, 3, 1)
    #    input = input.contiguous()
       print(input.is_contiguous(), input.shape)
       input = input.cuda()
       m = m.cuda()
    elif torch.__version__ == "1.3.0a0":
        torch.set_printoptions(10)
        import torch_mlu.core.mlu_model as ct
        ct.set_cnml_enabled(False)
        m = m.to(ct.mlu_device())
        input = input.to(ct.mlu_device())
    elif torch.__version__ == "1.3.1": # pytorch cpu
        torch.set_printoptions(10)
        pass
    out = m(input)
    out.backward(torch.ones_like(out))
    print(out.abs().sum().cpu())
