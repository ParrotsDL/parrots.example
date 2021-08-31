from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext',
                                 ['iou_overlap'])


def gpu_iou_overlap(b1, b2, mode='IoU', offset=1):
    """计算两个候选框 b1 和 b2 的 IoU / IoF / IoS。

    参数:
        - b1(Tensor): 框或者框集合，[N, >=4]
        - b2(Tensor): 候选框或框集合，[M, >=4]
        - mode(str): 重叠框的计算方法，{IoU, IoF, IoS} 分别计算交并 / 第一 / 第二 面积，默认 'IoU'
        - offset(int): 偏移量，默认 1
    """
    if b1.numel() == 0 or b2.numel() == 0:
        return b1.new_zeros((0,))

    flag = {'IoU': 0, 'IoF': 1, 'IoS': 2}[mode]

    assert b1.shape[1] >= 4 and b2.shape[1] >= 4
    assert b1.is_cuda and b2.is_cuda
    assert offset == 1 or offset == 0

    b1 = b1[:, :4].contiguous()
    b2 = b2[:, :4].contiguous()
    ious = b1.new_zeros((b1.shape[0], b2.shape[0]))
    ext_module.iou_overlap(b1, b2, ious, mode=flag, offset=offset)
    return ious
