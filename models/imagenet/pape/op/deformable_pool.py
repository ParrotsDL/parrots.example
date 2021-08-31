import torch
from torch import nn
from torch.autograd import Function

from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext',
                                 ['deform_psroi_pooling_cuda_forward',
                                  'deform_psroi_pooling_cuda_backward'])


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                data,
                rois,
                offset,
                spatial_scale,
                out_size,
                out_channels,
                no_trans,
                group_size=1,
                part_size=None,
                sample_per_part=4,
                trans_std=.0):

        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        ext_module.deform_psroi_pooling_cuda_forward(
            data, rois, offset, output, output_count,
            no_trans=ctx.no_trans,
            spatial_scale=ctx.spatial_scale,
            output_dim=ctx.out_channels,
            group_size=ctx.group_size,
            pooled_size=ctx.out_size,
            part_size=ctx.part_size,
            sample_per_part=ctx.sample_per_part,
            trans_std=ctx.trans_std)

        ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)

        ext_module.deform_psroi_pooling_cuda_backward(
            grad_output, data, rois, offset, output_count, grad_input,
            grad_offset,
            no_trans=ctx.no_trans,
            spatial_scale=ctx.spatial_scale,
            output_dim=ctx.out_channels,
            group_size=ctx.group_size,
            pooled_size=ctx.out_size,
            part_size=ctx.part_size,
            sample_per_part=ctx.sample_per_part,
            trans_std=ctx.trans_std)
        return (grad_input, grad_rois, grad_offset, None, None, None, None,
                None, None, None, None)


class DeformRoIPooling(nn.Module):
    """
    可变形卷积 RoI 池化。

    参数:
       - spaital_scale(float): 空间缩放因子
       - out_size(tuple): 输出空间大小
       - out_channels(int): 输出通道数
       - no_trans(bool): 是否转置
       - group_size(int): 组数，默认1
       - part_size(int): 分块大小，默认 None
       - sample_per_part(int): 每部分的采样点数，默认4
       - trans_std(float): 宽度偏移参数，默认 0.0

    输入:
       - data(Tensor): 输入
       - rois(Tensor): 候选框
       - offset(Tensor): 偏移量
    """

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return DeformRoIPoolingFunction.apply(
            data, rois, offset, self.spatial_scale, self.out_size,
            self.out_channels, self.no_trans, self.group_size, self.part_size,
            self.sample_per_part, self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):
    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_channels=1024):
        super(DeformRoIPoolingPack,
              self).__init__(spatial_scale, out_size, out_channels, no_trans,
                             group_size, part_size, sample_per_part, trans_std)

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return DeformRoIPoolingFunction.apply(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = DeformRoIPoolingFunction.apply(
                data, rois, offset, self.spatial_scale,
                self.out_size, self.out_channels, True,
                self.group_size, self.part_size,
                self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            return DeformRoIPoolingFunction.apply(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(
            spatial_scale, out_size, out_channels, no_trans, group_size,
            part_size, sample_per_part, trans_std)

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            self.mask_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 1),
                nn.Sigmoid())
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return DeformRoIPoolingFunction.apply(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = DeformRoIPoolingFunction.apply(
                data, rois, offset, self.spatial_scale,
                self.out_size, self.out_channels, True,
                self.group_size, self.part_size,
                self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size, self.out_size)
            return DeformRoIPoolingFunction.apply(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std) * mask
