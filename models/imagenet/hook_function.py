
import os
import torch
import pickle
import numpy as np
from termcolor import colored

### 更换版本记得改这里 ！！！！！！
CAMB_TORCH_VERSION = "1.6.0a0"
CAMB_PARROTS_VERSION = "parrots"
CPU_PYTORCH = "1.3.1"

code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')


'''
# 介绍
改类是进行精度比较的hook工具
首先在pytorch环境下运行
然后在parrots环境下运行
运行结束后会比较两次的结果

# 使用方法
hct = hookCompareTool()

m = AlexNet()
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
'''

class hookCompareTool():
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        
        self.torch_version = torch.__version__
        self.order_counter = 0
        self.data_dict = {}
        self.name_list_forward = []
        self.name_list_backward = []
        self.pkl_save_path = "data/camb_torch_hook.pkl"
        self.torch_checkpoint_path = "data/checkpoint.pth"
        self.torch_input_path = "data/input.pth"

        self.make_dirs(self.pkl_save_path)
        self.make_dirs(self.torch_checkpoint_path)
        self.make_dirs(self.torch_input_path)

        print("pytorch version:", code_green(torch.__version__))

    def make_dirs(self, path):
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    def insert(self, name, mm, tag, input, output):
        if "Sequential" in f"{mm}":
            return
        if name not in self.data_dict.keys():
            self.data_dict[name] = {}
        if tag == "forward":
            self.name_list_forward.append(name)
            self.data_dict[name]['forward'] = {"op": f"{mm}", "input": input, "output": output}
        else:
            self.name_list_backward.append(name)
            self.data_dict[name]['backward'] = {"op": f"{mm}", "input": output, "output": input}

    def pkl_save(self):
        with open(self.pkl_save_path, "wb") as file:
            pickle.dump(self.data_dict, file)
            print("save file at {}".format(self.pkl_save_path))

    def pkl_read(self):
        with open(self.pkl_save_path, "rb") as file:
            return pickle.load(file)

    def pkl_compare(self):
        camb_torch_dict = self.pkl_read()
        for name in self.name_list_forward:
            if not name in camb_torch_dict.keys() or not name in self.data_dict.keys():
                print(f"The key {code_grey(name)} does not match")
                continue
            print(f"Layer {code_yellow(name)} {code_yellow('forward')} {code_green(camb_torch_dict[name]['forward']['op'])}")
            print(f"input {code_magenta(camb_torch_dict[name]['forward']['input'])} vs {code_magenta(self.data_dict[name]['forward']['input'])}")
            print(f"output {code_cyan(camb_torch_dict[name]['forward']['output'])} vs {code_cyan(self.data_dict[name]['forward']['output'])}")
            print()

        for name in self.name_list_backward:
            if not name in camb_torch_dict.keys() or not name in self.data_dict.keys():
                print(f"The key {code_grey(name)} does not match")
                continue
            print(f"Layer {code_yellow(name)} {code_yellow('backward')} {code_green(camb_torch_dict[name]['backward']['op'])}")
            print(f"input {code_magenta(camb_torch_dict[name]['backward']['input'])} vs {code_magenta(self.data_dict[name]['backward']['input'])}")
            print(f"output {code_cyan(camb_torch_dict[name]['backward']['output'])} vs {code_cyan(self.data_dict[name]['backward']['output'])}")
            print()

    # 保存并且打印比较结果
    def save_and_compare_hook(self):
        if self.torch_version == CAMB_TORCH_VERSION:
            self.pkl_save()
        elif self.torch_version == CAMB_PARROTS_VERSION:
            self.pkl_compare()

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

    def get_data_numpy(self, data):
        assert self.reduction in ['mean', 'sum', 'shape', 'all']
        if isinstance(data, (torch.Tensor)):
            npdata = data.clone().detach().cpu().numpy()
            if self.reduction == 'mean':
                return np.mean(npdata)
            elif self.reduction == 'sum':
                return np.sum(npdata)
            elif self.reduction == 'shape':
                return npdata.shape
            else:
                return npdata.reshape(-1)[-5:], np.mean(npdata), npdata.shape
        elif isinstance(data, dict):
            return {k: self.get_data_numpy(v) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return data.__class__(self.get_data_numpy(v) for v in data)
        else:
            return data

    def hook(self, name, mm, tag='forward'):
        hook_func = self.get_data_numpy
        def inner_hook(m, input, output):
            self.insert(name, mm, tag, hook_func(input), hook_func(output))
            if tag == 'forward':
                print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(input), hook_func(output)))
            else:
                print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(output), hook_func(input)))
        return inner_hook

    def save_and_load(self, input, model):
        # pytorch环境下保存模型参数和输入
        if self.torch_version != CAMB_PARROTS_VERSION:
            torch.save(model.state_dict(), self.torch_checkpoint_path)
            torch.save(input, self.torch_input_path)
        # parrots环境下固定模型参数和输入
        else:
            input = torch.load(self.torch_input_path)
            model.load_state_dict(torch.load(self.torch_checkpoint_path))

        return input, model

    def to_cuda(self,input, model):
        if self.torch_version == CAMB_PARROTS_VERSION:
            # torch.cuda.set_device(4)
            model = model.to_memory_format(torch.channels_last)
            model = model.cuda()
            # input = input.contiguous(torch.channels_last)
            input = input.cuda()
            input = input.contiguous(torch.channels_last)
        else:
            torch.set_printoptions(10)
            import torch_mlu.core.mlu_model as ct
            ct.set_device(4)
            ct.set_cnml_enabled(False)
            model = model.to(ct.mlu_device())
            input = input.to(ct.mlu_device())
        
        return input, model
