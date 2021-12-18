import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")
import imagenet.hook as hook

from yolov3.models.yolo import Detect, Model, parse_model
from yolov3.utils.loss import ComputeLoss

torch_version = torch.__version__
hook.CAMB_TORCH_VERSION = "1.6.0a0+ab70945" # no use
hook.CPU_PYTORCH_VERSION = "1.3.1+cpu"

if __name__ == "__main__":
    hc = hook.hookCompare()
    
    yaml_path = "./yolov3.yaml"
    model = Model(yaml_path, ch=3)
    in_size = 256
    input = torch.randn(1, 3, in_size, in_size)
    # input = torch.randn(2, 32, 5, 5)
    compute_loss = ComputeLoss(model)
    target = torch.load("./data/yolov3_targets.pth")
    

    # print(model)
    
    if torch_version == hook.CPU_PYTORCH_VERSION:
        from torchsummary import summary
        # summary(model, input_size=(3, in_size, in_size))
        # summary(model, input_size=(32, 5, 5))
        
    
    for name, mm in model.named_modules():
        # print(hook.code_yellow(name), "---", hook.code_green(mm))
        mm.register_forward_hook(hc.hook(name, mm))
        mm.register_backward_hook(hc.hook(name, mm, tag='backward'))
        
    input, model = hc.save_and_load2(input, model)
    model.train()
    
    if torch_version == hook.CAMB_PARROTS_VERSION: # parrots
        model = model.to_memory_format(torch.channels_last)
        model = model.cuda()
        input = input.cuda()
        target = target.cuda()
        if input.ndims == 4:
            input = input.contiguous(torch.channels_last)
        # target = target.int().cuda()
        # criterion = criterion.cuda()
        from torch.cuda import amp
        with amp.autocast(enabled=True):
            output = model(input)
            loss, loos_item = compute_loss(output, target)
        pass
    elif torch_version == hook.CPU_PYTORCH_VERSION: # cpu pytorch
        output = model(input)
        pass
    else: # camb pytorch
        torch.set_printoptions(10)
        import torch_mlu.core.mlu_model as ct
        ct.set_cnml_enabled(False)
        ct.set_quantized_bitwidth(16)
        model = model.to(ct.mlu_device())
        input = input.to(ct.mlu_device())
        target = target.to(ct.mlu_device())
        criterion = criterion.to(ct.mlu_device())
        HalfModel(model)
        input = input.half()
        pass

    hc.save_and_compare_hook()
    
    print(output.__len__(), output[0].shape, output[0].mean())