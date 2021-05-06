import torch
import models
import torch.nn.functional as F
import torch.nn as nn
# import models
import models.mobile_v2_min
import numpy as np

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

def hook(name, tag='forward'):
    hook_func = get_data
    def inner_hook(m, input, output):
         print("Layer {} {}:\n input {} \n output {}\n".format(name, tag, hook_func(input), hook_func(output)))
        #  print("Layer {} {}:\n input {}\n output {}\n".format(name, tag, get_data2(input), get_data2(output)))
    return inner_hook

class mv2(nn.Module):
    def __init__(self, num_classes=1000):
        super(mv2, self).__init__()

        # self.conv = nn.Conv2d(96, 96, 3, 1, 1, groups=1)
        self.conv = nn.Conv2d(32, 32, kernel_size=(3, 3), 
            stride=(1, 1), padding=(1, 1), groups=32, bias=False)

    def forward(self, x):
        x = self.conv(x)
        
        print(x.shape)
        return x

if __name__=="__main__":
    
    # m = models.mobile_v2()
    # m = models.mobile_v2_min.mobile_v2()
    m = mv2()

    # input = torch.randn(2, 3, 224, 224, requires_grad=True)
    input = torch.ones(2, 32, 224, 224, requires_grad=True)
    
    #进行hook注册访问每一层的forward和backward输入输出
    for name, mm in m.named_modules():
        print(name, "---", mm)
        mm.register_forward_hook(hook(name))
        # mm.register_backward_hook(hook(name, tag='backward'))

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
    else:
        torch.set_printoptions(6)
        import torch_mlu.core.mlu_model as ct
        ct.set_cnml_enabled(False)
        m = m.to(ct.mlu_device())
        input = input.to(ct.mlu_device())
    print(get_data(input))
    out = m(input)
    out.backward(torch.ones_like(out))
    print(out.abs().sum().cpu())
