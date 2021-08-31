import torch
from torch import nn

from ..op import (ModulatedDeformConv, DeformRoIPoolingPack,
                  ModulatedDeformRoIPoolingPack, SyncBatchNorm2d)
from ..op import DeformableConv
from ..op import naive_nms
from ..op import RoIPool as PAPERoIPool
from ..op import RoIAlignPool
from ..op import soft_nms as pape_soft_nms
from ..op import SigmoidFocalLossFunction


class DeformConv(DeformableConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        assert not bias
        super(DeformConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            num_deformable_groups=deformable_groups)


def nms(dets, iou_thr, device_id=None):
    assert isinstance(dets, torch.Tensor)

    # sort dets by scores
    scores = dets[:, -1]
    _, score_rank_idx = scores.sort(descending=True)
    dets = dets[score_rank_idx]

    if dets.shape[0] == 0:
        inds = []
    else:
        inds = naive_nms(dets, iou_thr)

    inds = dets.new_tensor(inds, dtype=torch.long)
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    assert isinstance(dets, torch.Tensor)

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))

    # sort dets by scores
    scores = dets[:, -1]
    _, score_rank_idx = scores.sort(descending=True)
    dets = dets[score_rank_idx]

    new_dets, inds = pape_soft_nms(
        dets.cpu(),
        Nt=iou_thr,
        method=method_codes[method],
        sigma=sigma,
        thresh=min_score)

    return new_dets, inds


class RoIAlign(RoIAlignPool):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        if isinstance(out_size, int):
            pooled_h = out_size
            pooled_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            pooled_h, pooled_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        super(RoIAlign, self).__init__(pooled_h, pooled_w, sample_num)
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.stride = 1 / spatial_scale

    def forward(self, features, rois):
        return super(RoIAlign, self).forward(rois, features, self.stride)


class RoIPool(PAPERoIPool):

    def __init__(self, out_size, spatial_scale):
        if isinstance(out_size, int):
            pooled_h = out_size
            pooled_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            pooled_h, pooled_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        super(RoIPool, self).__init__(pooled_h, pooled_w)
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.stride = 1 / float(spatial_scale)

    def forward(self, features, rois):
        return super(RoIPool, self).forward(rois, features, self.stride)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


def sigmoid_focal_loss(input, target, gamma=2.0, alpha=0.25):
    f = SigmoidFocalLossFunction(gamma, alpha, input.size(-1))
    return f(input, target.int(), target.new_tensor([1.]))


__all__ = [
    'RoIPool', 'RoIAlign', 'nms', 'soft_nms', 'SyncBatchNorm2d', 'DeformConv',
    'ModulatedDeformConv', 'DeformRoIPoolingPack', 'ContextBlock',
    'ModulatedDeformRoIPoolingPack', 'sigmoid_focal_loss'
]
