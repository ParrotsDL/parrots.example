import logging

import torch
from torch.autograd import Function

from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext',
                                 ['roi_align_forward',
                                  'roi_align_backward'])

logger = logging.getLogger('global')


class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio):
        return g.op(
            "RoiAlign",
            features,
            rois,
            spatial_scale_f=spatial_scale,
            pooled_width_i=pooled_w,
            pooled_height_i=pooled_h,
            sample_num_i=sampling_ratio)

    @staticmethod
    def forward(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio):
        self.save_for_backward(features, rois)
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, pooled_h, pooled_w).zero_()
        assert features.is_contiguous() and rois.is_contiguous()

        if not features.is_cuda:
            logger.warning('---CPU version of RoIAlignPooling is a dummpy function, which is used to support tocaffe')

        ext_module.roi_align_forward(
            features,
            rois,
            output,
            aligned_height=pooled_h,
            aligned_width=pooled_w,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        grad_output = grad_output.data
        feature, rois = self.saved_tensors

        batch_size, num_channels, data_height, data_width = feature.shape
        grad_input = feature.new(batch_size, num_channels, data_height, data_width).zero_()

        assert grad_output.is_contiguous()
        assert rois.is_contiguous()

        ext_module.roi_align_backward(
            grad_output,
            rois,
            grad_input,
            aligned_height=self.pooled_h,
            aligned_width=self.pooled_w,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio)
        return grad_input, None, None, None, None, None


class RoIAlignPool(torch.nn.Module):
    """
    进行 RoI 的池化对齐操作，第一次描述在论文 `Mask_RCNN <https://arxiv.org/pdf/1703.06870.pdf>`_。


    参数:
        - pooled_h(int): 池化输出的高
        - pooled_w(int): 池化输出的宽
        - sampling_ratio(int): 网格的采样点数
        - spatial_ratio(float): 输入到输出的缩放因子
    输入:
        - rois(Tensor): 候选框集合，[N, >=5] (batch_idx, x1, y1, x2, y2)
        - features(Tensor): 特征向量，[N, C, H, W]
        - stride(int): 池化步长

    .. note::
        - rois 必须是 N*5 维
        - 在混合精度训练时，featues 的类型可能是 fp16, 但是 rois 则可能不是
        - 在 tensor 传入 C 代码之前必须进行 contiguous
        - spatial_scale 在进行 RoIAlignPool 初始化时会被弃用，将该参数作为 forward 的 stride 参数能够更加灵活
    """

    def __init__(self, pooled_h, pooled_w, sampling_ratio, spatial_scale=None):
        super(RoIAlignPool, self).__init__()
        self.pooled_w = int(pooled_w)
        self.pooled_h = int(pooled_h)
        self.sampling_ratio = int(sampling_ratio)

        if spatial_scale is not None:
            logger.warning('spatial_scale is deprecated when initializing RoIAlignPool'
                           'we move spatial_scale to forward arguments `stride` for flexiability')

    def forward(self, rois, feature, stride):
        rois = rois[:, :5].contiguous().to(dtype=feature.dtype)
        feature = feature.contiguous()
        assert rois.shape[1] == 5, rois.shape
        spatial_scale = 1.0 / stride
        return RoIAlignFunction.apply(feature, rois, self.pooled_h,
                                      self.pooled_w, spatial_scale,
                                      self.sampling_ratio)

    def __repr__(self):
        s = '{name} ({pooled_h}, {pooled_w}, {sampling_ratio})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        pooled_h = pooled_w = params['pool_size']
        sampling_ratio = params['sampling_ratio']
        return cls(pooled_h, pooled_w, sampling_ratio)
