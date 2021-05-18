import torch
import models
import torch.nn.functional as F
import torch.nn as nn
import models.inception_v3_min
import numpy as np
from termcolor import colored
import pickle
import pdb

PYTORCH_VERSION = torch.__version__
CAMB_TORCH = "1.3.0a0"
PYTORCH = "1.3.1"
PARROTS = "parrots"

code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')

class hookCompareTool():
    def __init__(self):
        self.order_counter = 0
        self.data_dict = {}
        self.name_list_forward = []
        self.name_list_backward = []

    def insert(self, name, mm, tag, input, output):
        if name not in self.data_dict.keys():
            self.data_dict[name] = {}
        if tag == "forward":
            self.name_list_forward.append(name)
            self.data_dict[name]['forward'] = {"op": f"{mm}", "input": input, "output": output}
        else:
            self.name_list_backward.append(name)
            self.data_dict[name]['backward'] = {"op": f"{mm}", "input": output, "output": input}

    def pkl_save(self, path='data/camb_torch_hook.pkl'):
        with open(path, "wb") as file:
            pickle.dump(self.data_dict, file)

    def pkl_read(self, path='data/camb_torch_hook.pkl'):
        with open(path, "rb") as file:
            return pickle.load(file)

    def pkl_compare(self):
        camb_torch_dict = self.pkl_read()
        for name in self.name_list_forward:
            if not name in camb_torch_dict.keys() or not name in self.data_dict.keys():
                print(f"The key {name} does not match")
                continue
            print(f"Layer {code_yellow(name)} {code_yellow('forward')} {code_green(camb_torch_dict[name]['forward']['op'])}")
            print(f"input {code_magenta(camb_torch_dict[name]['forward']['input'])} vs {code_magenta(self.data_dict[name]['forward']['input'])}")
            print(f"output {code_cyan(camb_torch_dict[name]['forward']['output'])} vs {code_cyan(self.data_dict[name]['forward']['output'])}")
            print()

        for name in self.name_list_backward:
            if not name in camb_torch_dict.keys() or not name in self.data_dict.keys():
                print(f"The key {code_red(name)} does not match")
                continue
            print(f"Layer {code_yellow(name)} {code_yellow('backward')} {code_green(camb_torch_dict[name]['backward']['op'])}")
            print(f"input {code_magenta(camb_torch_dict[name]['backward']['input'])} vs {code_magenta(self.data_dict[name]['backward']['input'])}")
            print(f"output {code_cyan(camb_torch_dict[name]['backward']['output'])} vs {code_cyan(self.data_dict[name]['backward']['output'])}")
            print()

    def get_data(self, data, reduction='mean'):
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
            return {k: self.get_data(v, reduction) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return data.__class__(self.get_data(v, reduction) for v in data)
        else:
            return data

    def get_data_numpy(self, data, reduction='mean'):
        assert reduction in ['mean', 'sum', 'shape', 'all']
        if isinstance(data, (torch.Tensor)):
            npdata = data.clone().detach().cpu().numpy()
            if reduction == 'mean':
                return np.mean(npdata)
            elif reduction == 'sum':
                return np.sum(npdata)
            elif reduction == 'shape':
                return npdata.shape
            else:
                return npdata.reshape(-1)[-5:], np.sum(npdata)
        elif isinstance(data, dict):
            return {k: self.get_data_numpy(v, reduction) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return data.__class__(self.get_data_numpy(v, reduction) for v in data)
        else:
            return data

    def hook(self, name, mm, tag='forward'):
        hook_func = self.get_data_numpy
        def inner_hook(m, input, output):
            self.insert(name, mm, tag, hook_func(input), hook_func(output))
            # if tag == 'forward':
            #     print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(input), hook_func(output)))
            # else:
            #     print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(output), hook_func(input)))
        return inner_hook

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        x = torch.cat(outputs, 1)
        return x

class incv3(nn.Module):
    def __init__(self, num_classes=1000):
        super(incv3, self).__init__()

        self.Mixed_5b = InceptionA(288, pool_features=64)
        self.Mixed_5c = InceptionA(288, pool_features=64)


    def forward(self, x):

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        
        return x


if __name__=="__main__":

    hct = hookCompareTool()
    
    m = models.inception_v3()
    # m = models.inception_v3_min.inception_v3()
    # m = incv3()

    # input = torch.randn(2, 288, 35, 35, requires_grad=True)
    input = torch.randn(2, 3, 299, 299, requires_grad=True)
    
    #进行hook注册访问每一层的forward和backward输入输出
    for name, mm in m.named_modules():
        print(code_blue(name), "---", code_blue(mm))
        mm.register_forward_hook(hct.hook(name, mm))
        mm.register_backward_hook(hct.hook(name, mm, tag='backward'))

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
        # torch.set_printoptions(10)
        import torch_mlu.core.mlu_model as ct
        ct.set_cnml_enabled(False)
        m = m.to(ct.mlu_device())
        input = input.to(ct.mlu_device())
    elif torch.__version__ == "1.3.1": # pytorch cpu
        pass

    out = m(input)
    out.backward(torch.ones_like(out))

    if PYTORCH_VERSION == CAMB_TORCH:
        hct.pkl_save()
    elif PYTORCH_VERSION == PARROTS:
        hct.pkl_compare()

    print(out.shape, out.abs().sum().cpu())