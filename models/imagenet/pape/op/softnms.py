import numpy as np
import torch
from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext',
                                 ['softnms'])


def soft_nms(dets, sigma=0.5, Nt=0.3, thresh=0.001, method=0):
    """
    nms 的三种实现方法。

    输入:
        - dets(Tensor): 候选框集合
        - sigma(float): gaussian 方法使用的参数
        - Nt(float): 线性方法使用的参数
        - thresh(float): 朴素方法使用的参数
        - method(int,[0,1,2]): 计算方法选择，0 代表朴素方法，１代表线性方法，２代表高斯方法

    输出：
        筛选出的候选框

    .. note::
    　　目前仅支持 CPU 端的计算

    """
    assert dets.shape[1] == 5
    # assert not dets.is_cuda
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = ((x2 - x1 + 1) * (y2 - y1 + 1)).float()
    order = torch.from_numpy(np.arange(dets.shape[0])).long()
    # keep = torch.LongTensor(dets.shape[0])
    num_out = torch.LongTensor(1)
    bboxes = dets.clone().float()
    ext_module.softnms(
        bboxes,
        areas,
        order,
        num_out,
        sigma=sigma,
        n_thresh=Nt,
        overlap_thresh=thresh,
        method=method)
    ret_bboxes = bboxes[order[:num_out[0]]]
    _, inds = torch.sort(ret_bboxes, dim=0, descending=True)
    ret_bboxes = ret_bboxes[inds[:, 4]]
    return ret_bboxes.contiguous().to(
           device=dets.device), order[:num_out[0]].contiguous().to(
           device=dets.device)
