import torch
import math
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..utils import ext_loader
ext_module = ext_loader.load_ext('op_ext',
                                 ['modulated_deform_conv_cuda_forward',
                                  'modulated_deform_conv_cuda_backward'])


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                offset,
                mask,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_cuda_forward(
            input, weight, bias, ctx._bufs[0],
            offset, mask, output, ctx._bufs[1],
            kernel_h=weight.shape[2], kernel_w=weight.shape[3],
            stride_h=ctx.stride, stride_w=ctx.stride,
            pad_h=ctx.padding, pad_w=ctx.padding,
            dilation_h=ctx.dilation, dilation_w=ctx.dilation, group=ctx.groups,
            deformable_group=ctx.deformable_groups, with_bias=ctx.with_bias)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        ext_module.modulated_deform_conv_cuda_backward(
            input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1],
            grad_input, grad_weight,
            grad_bias, grad_offset,
            grad_mask, grad_output,
            kernel_h=weight.shape[2], kernel_w=weight.shape[3],
            stride_h=ctx.stride, stride_w=ctx.stride,
            pad_h=ctx.padding, pad_w=ctx.padding,
            dilation_h=ctx.dilation, dilation_w=ctx.dilation, group=ctx.groups,
            deformable_group=ctx.deformable_groups, with_bias=ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


class ModulatedDeformConv(nn.Module):
    """
    DeformableConv 的改进版本。

    参数：
    　　 - in_channels(int): 输入通道数
    　　 - out_channels(int): 输出通道数
    　　 - kernel_size(int): 卷积核大小
    　　 - stride(int): 卷积核步长，默认１
    　　 - padding(int): 填充，默认１
    　　 - groups(int): 组数，默认１
    　　 - deformable_groups(int): 可变性卷积组数，通道数的分组，默认１
    　　 - bias(bool): 是否添加偏置，默认 True

    输入：
    　　 - input(Tensor): 特征向量
    　　 - offset(Tensor): 偏移量
         - mask(Tensor): 掩码
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return ModulatedDeformConvFunction.apply(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConvPack, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, deformable_groups, bias)

        self.conv_offset_mask = nn.Conv2d(
            self.in_channels // self.groups,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        # o1, o2, mask = torch.chunk(out, 3, dim=1)
        o1, o2, mask = torch.split(out, out.shape[1] // 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return ModulatedDeformConvFunction.apply(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)
