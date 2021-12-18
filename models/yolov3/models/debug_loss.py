
import torch
import torch.nn.functional as F
import torch.nn as nn


import sys
sys.path.append("..")
sys.path.append("../..")
import imagenet.hook as hook

from yolov3.models.yolo import Detect, Model, parse_model
from yolov3.utils.loss import ComputeLoss

torch_version = torch.__version__
hook.CAMB_TORCH_VERSION = "1.6.0a0+ab70945" # no use
hook.CPU_PYTORCH_VERSION = "1.3.1+cpu"

def printOutput(output):
    print("output=======") 
    print(output[0].stride())
    output = [x.contiguous().float().detach().cpu() for x in output]
    [print(x.shape) for x in output]
    [print(x.abs().mean()) for x in output]
    [print(x.abs().sum()) for x in output]
    [print(x.reshape(-1)[:10]) for x in output]
    print("output======= \n") 

if __name__ == "__main__":
    hc = hook.hookCompare()
    
    yaml_path = "./yolov3.yaml"
    model = Model(yaml_path, ch=3)
    
    input = torch.load("./data/yolov3_input.pth")
    input = input.float()
    target = torch.load("./data/yolov3_targets.pth")
    target = target.float()

    model.train()
    
    if torch_version == hook.CAMB_PARROTS_VERSION: # parrots
        output = torch.load("./data/yolov3_output.pth")
        printOutput(output)
        output = [x.cuda().half() for x in output]

        # input = input.cuda().contiguous(torch.channels_last)
        model = model.cuda().to_memory_format(torch.channels_last)
        compute_loss = ComputeLoss(model)
        target = target.cuda()
        # if input.ndims == 4:
        #     input = input.contiguous(torch.channels_last)
        from torch.cuda import amp
        with amp.autocast():
            loss, loss_item = compute_loss(output, target)
    elif torch_version == hook.CPU_PYTORCH_VERSION: # cpu pytorch
        output = model(input)
        printOutput(output)
        torch.save(output, "./data/yolov3_output.pth")
        compute_loss = ComputeLoss(model)
        loss, loss_item = compute_loss(output, target)
        pass
    else: # camb pytorch
        # torch.set_printoptions(10)
        # import torch_mlu.core.mlu_model as ct
        # ct.set_cnml_enabled(False)
        # ct.set_quantized_bitwidth(16)
        # model = model.to(ct.mlu_device())
        # input = input.to(ct.mlu_device())
        # target = target.to(ct.mlu_device())
        # criterion = criterion.to(ct.mlu_device())
        # HalfModel(model)
        # input = input.half()
        pass

    # output = [x.float() for x in output]
    # print("output:", output.__len__(), output[0].shape, output[0].abs().mean(), output[0].mean().sum())
    # out = output[0].contiguous()
    # out_view = out.reshape(-1)
    # print("view:", out_view[:10])
    
    # # print("cpu output:", cpu_out.__len__(), cpu_out[0].shape, cpu_out[0].sum())
    # # cpu_out_view = cpu_out[0].reshape(-1)
    # # print("cpu_view:", cpu_out_view[:10])

    # printOutput(output)
    print("loss:", loss, loss.item())