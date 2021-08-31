import torch


def to_half(data):
    """
    将数据转化为半精度
    """
    if isinstance(data, list):
        res = []
        for t in data:
            res.append(to_half(t))
        return res
    elif isinstance(data, tuple):
        res = []
        for t in data:
            res.append(to_half(t))
        return tuple(res)
    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = to_half(v)
        return res
    elif isinstance(data, (torch.cuda.FloatTensor, torch.FloatTensor)):
        return data.half()
    else:
        return data


def to_float(data):
    """
    将数据转化为浮点数
    """
    if isinstance(data, list):
        res = []
        for t in data:
            res.append(to_float(t))
        return res
    elif isinstance(data, tuple):
        res = []
        for t in data:
            res.append(to_float(t))
        return tuple(res)
    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = to_float(v)
        return res
    elif isinstance(data, (torch.cuda.HalfTensor, torch.HalfTensor)):
        return data.float()
    else:
        return data


def to_cuda(data):
    if isinstance(data, list):
        res = []
        for t in data:
            res.append(to_cuda(t))
        return res
    elif isinstance(data, tuple):
        res = []
        for t in data:
            res.append(to_cuda(t))
        return tuple(res)
    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = to_cuda(v)
        return res
    elif isinstance(data, (torch.Tensor)):
        return data.cuda()
    else:
        return data


def to_cpu(data):
    if isinstance(data, list):
        res = []
        for t in data:
            res.append(to_cpu(t))
        return res
    elif isinstance(data, tuple):
        res = []
        for t in data:
            res.append(to_cpu(t))
        return tuple(res)
    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = to_cpu(v)
        return res
    elif isinstance(data, (torch.Tensor)):
        return data.cpu()
    else:
        return data


def has_inf_or_nan(x):
    return torch.isnan(x).any().item() != 0 or torch.isinf(x).any().item() != 0
