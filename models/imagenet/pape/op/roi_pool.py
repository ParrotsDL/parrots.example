import logging

import torch
from torch.autograd import Function

from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext',
                                 ['roi_pooling_forward',
                                  'roi_pooling_backward'])

logger = logging.getLogger('global')


class RoIPoolFunction(Function):
    @staticmethod
    def symbolic(g, features, rois, pooled_h, pooled_w, spatial_scale):
        return g.op(
            "RoiPool",
            features,
            rois,
            spatial_scale_f=spatial_scale,
            pooled_width_i=pooled_w,
            pooled_height_i=pooled_h)

    @staticmethod
    def forward(self, features, rois, pooled_height, pooled_width, spatial_scale):
        self.save_for_backward(features, rois)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, pooled_height, pooled_width).zero_()
        self.argmax = features.new(num_rois, num_channels, pooled_height, pooled_width).zero_().int()

        assert features.is_contiguous()
        assert rois.is_contiguous()

        if not features.is_cuda:
            logger.warning('---CPU version of RoIPooling is a dummpy function, which is used to support tocaffe')

        ext_module.roi_pooling_forward(
            features,
            rois,
            output,
            self.argmax,
            pooled_height=self.pooled_height,
            pooled_width=self.pooled_width,
            spatial_scale=self.spatial_scale)

        return output

    @staticmethod
    def backward(self, grad_output):
        feature, rois = self.saved_tensors
        batch_size, num_channels, data_height, data_width = feature.size()
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        assert grad_output.is_contiguous()
        assert rois.is_contiguous()

        ext_module.roi_pooling_backward(
            grad_output,
            rois,
            self.argmax,
            grad_input,
            pooled_height=self.pooled_height,
            pooled_width=self.pooled_width,
            spatial_scale=self.spatial_scale)

        return grad_input, None, None, None, None


class RoIPool(torch.nn.Module):
    """
    候选框的池化操作，输出特定大小的特征图。

    参数：
        - pooled_h(int): 输出图的高
        - pooled_w(int): 输出图的宽
        - spatial_scale(float): 缩放因子，默认 None

    输入：
        - rois(Tensor): 候选框集合 [N, >=5](batch_idx, x1, y1, x2, y2)
        - features(Tensor): 特征图，feature map
        - stride(int): 池化操作步长

    .. note::
        - rois 必须是 N*5 维
        - 在混合精度训练时，featues 的类型可能是 fp16, 但是 rois 则可能不是
        - 在 tensor 传入 C 代码之前必须进行 contiguous 操作
        - spatial_scale 在进行 RoIPool 初始化时会被弃用，将该参数作为 forward 的 stride 参数能够更加灵活

    """
    def __init__(self, pooled_h, pooled_w, spatial_scale=None):
        super(RoIPool, self).__init__()
        self.pooled_h = int(pooled_h)
        self.pooled_w = int(pooled_w)
        if spatial_scale is not None:
            logger.warning('spatial_scale is deprecated when initializing'
                           'RoIPool we move spatial_scale to forward '
                           'arguments `stride` for flexiability')

    def forward(self, rois, feature, stride):
        """
        Arguments:
            rois: [N, >=5] (batch_idx, x1, y1, x2, y2)

        Notes:
            1. rois must be N*5 dim
            2. in fp16 mode, feature.dtype is fp16, but rois.dtype may not
            3. tensor must be contiguous before passing to the C code
        """
        rois = rois[:, :5].contiguous().to(dtype=feature.dtype)
        feature = feature.contiguous()
        assert rois.shape[1] == 5, rois.shape
        spatial_scale = 1.0 / stride
        return RoIPoolFunction.apply(feature, rois, self.pooled_h,
                                     self.pooled_w, spatial_scale)

    def __repr__(self):
        s = '{name} ({pooled_h}, {pooled_w})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @classmethod
    def from_params(cls, params):
        pooled_h = pooled_w = params['pool_size']
        return cls(pooled_h, pooled_w)
