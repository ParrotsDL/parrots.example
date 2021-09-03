import torch
import torch.nn as nn
import math
import models

import numpy as np
import torch
import copy
from pape.half.half_model import HalfModel
from termcolor import colored
import math

code_yellow = lambda x: colored(x, 'yellow')
code_red = lambda x: colored(x, 'red')
code_green = lambda x: colored(x, 'green')
code_blue = lambda x: colored(x, 'blue')
code_grey = lambda x: colored(x, 'grey')
code_magenta = lambda x: colored(x, 'magenta')
code_cyan = lambda x: colored(x, 'cyan')

def get_data_numpy(data, reduction='sum'):
        if isinstance(data, (torch.Tensor)):
            npdata = data.float().clone().detach().cpu().numpy()
            if reduction == 'mean':
                return np.mean(npdata)
            elif reduction == 'sum':
                return np.sum(npdata)
            elif reduction == 'shape':
                return npdata.shape
            else:
                return npdata.reshape(-1)[-5:], np.mean(npdata), npdata.shape
        elif isinstance(data, dict):
            return {k: get_data_numpy(v) for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            return data.__class__(get_data_numpy(v) for v in data)
        else:
            return data

def insert(name, mm, tag, input, output, dict):
    if "Sequential" in f"{mm}":
            return
    if name not in dict.keys():
        dict[name] = {}
    if tag == "forward":
        dict[name]['forward'] = {"op": f"{mm}", "input": input, "output": output}
    else:
        dict[name]['backward'] = {"op": f"{mm}", "input": output, "output": input}

def hook(name, mm, dict, tag='forward'):
    hook_func = get_data_numpy
    def inner_hook(m, input, output):
        insert(name, mm, tag, hook_func(input), hook_func(output), dict)
        if tag == 'forward':
            print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(input), hook_func(output)))
        else:
            print("Layer {} {} {}:\n input {} \n output {}\n".format(code_yellow(name), code_yellow(tag), code_green(mm), hook_func(output), hook_func(input)))
    return inner_hook

def compare(dict1, dict2):
    k2 = 'forward'
    for name in dict1.keys():
        if not name in dict2.keys(): continue
        print(f"Layer {code_yellow(name)} {code_red(k2)} {code_green(dict1[name][k2]['op'])}")
        print(f"input {code_magenta(dict1[name][k2]['input'])} vs {code_magenta(dict2[name][k2]['input'])}")
        print(f"output {code_cyan(dict1[name][k2]['output'])} vs {code_cyan(dict2[name][k2]['output'])}")
        print()
    # @TODO: 倒序输出
    k2 = 'backward'
    k_list = list(dict1.keys())
    k_list.reverse()
    for name in k_list:
        if not name in dict2.keys(): continue
        print(f"Layer {code_yellow(name)} {code_red(k2)} {code_green(dict1[name][k2]['op'])}")
        print(f"input {code_magenta(dict1[name][k2]['input'])} vs {code_magenta(dict2[name][k2]['input'])}")
        print(f"output {code_cyan(dict1[name][k2]['output'])} vs {code_cyan(dict2[name][k2]['output'])}")
        print()

HALF = True
# HALF = False

fp32_dict = {}
fp16_dict = {}

for i in range(1):
    model = models.resnet50()
    model_half = copy.deepcopy(model)
    
    model = model.to_memory_format(torch.channels_last).cuda()
    model_half = model_half.to_memory_format(torch.channels_last).cuda()

    # register hook
    for name, mm in model.named_modules():
        mm.register_forward_hook(hook(name, mm, fp32_dict, tag='forward'))
        mm.register_backward_hook(hook(name, mm, fp32_dict, tag='backward'))

    # for name, mm in model_half.named_modules():
    #     mm.register_forward_hook(hook(name, mm, fp16_dict, tag='forward'))
    #     mm.register_backward_hook(hook(name, mm, fp16_dict, tag='backward'))


    input = torch.randn(2, 3, 224, 224, requires_grad=True)
    input_half = input.clone().detach()

    input = input.contiguous(torch.channels_last).cuda()
    input_half = input_half.contiguous(torch.channels_last).cuda()
    
    # model_half = model_half.half()
    model_half = HalfModel(model_half)
    input_half = input_half.half()


    out = model(input)
    out_half = model_half(input_half)

    
    out.backward(torch.ones_like(out))
    out_half.backward(torch.ones_like(out_half))

    compare(fp32_dict, fp16_dict)

    print(out)
    print(out_half)
    out_half = out_half.float()
    print("diff", out - out_half)
