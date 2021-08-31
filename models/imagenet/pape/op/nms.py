import numpy as np
import torch
from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext', ['nms'])


def naive_nms(dets, thresh, offset=1):
    """
    朴素版本的 nms。

    输入:
        - dets(Tensor): 候选框集合
        - tresh(float): 阈值
        - offset(int): 偏移

    输出:
        LongTensor，筛选出的候选框集合
    """
    assert dets.shape[1] == 5
    assert offset in (0, 1)
    keep = torch.LongTensor(dets.shape[0])
    num_out = torch.LongTensor(1)
    if torch.cuda.is_available():
        areas = torch.empty(0)
        order = torch.empty(0)
        ext_module.nms(dets.cuda().float(), order, areas, keep, num_out, nms_overlap_thresh=thresh, offset=offset)
    else:
        dets = dets.cpu()
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        order = torch.from_numpy(np.arange(dets.shape[0])).long()
        ext_module.nms(dets.float(), order, areas, keep, num_out, nms_overlap_thresh=thresh, offset=offset)
    return keep[:num_out[0]].contiguous().to(device=dets.device)
