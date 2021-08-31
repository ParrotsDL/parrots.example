import torch
import parrots.nn.functional as F
from enum import Enum
from functools import partial
from torch import nn
from types import MethodType

from pape.utils.basemodel import PapeModel

from .q_utils import get_tensor_max_abs, get_tensor_avg_max_abs, \
    get_tensor_mean_n_stds_max_abs, get_tensor_avg_min_max, \
    get_tensor_mean_n_stds_min_max, symmetric_linear_quantization_params, \
    asymmetric_linear_quantization_params, get_tensor_min_max

from parrots.darray import quantizer as qt
from .conv_fuse import Conv_BN


FP_PREFIX = 'float_'


class LinearQuantMode(Enum):
    SYMMETRIC = 1
    ASYMMETRIC_UNSIGNED = 2
    ASYMMETRIC_SIGNED = 3


class ClipMode(Enum):
    NONE = 0
    AVG = 1
    N_STD = 2


def _get_saturation_fn(quant_mode, clip_mode, num_stds):
    if quant_mode == LinearQuantMode.SYMMETRIC:
        fns = {ClipMode.NONE: get_tensor_max_abs,
               ClipMode.AVG: get_tensor_avg_max_abs,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_max_abs, n_stds=num_stds)}
    else:  # Asymmetric mode
        fns = {ClipMode.NONE: get_tensor_min_max,
               ClipMode.AVG: get_tensor_avg_min_max,
               ClipMode.N_STD: partial(get_tensor_mean_n_stds_min_max, n_stds=num_stds)}
    return fns[clip_mode]


def _get_quant_params_from_tensor(tensor, num_bits, mode, clip=ClipMode.NONE, num_stds=None):
    if clip == ClipMode.N_STD:
        if num_stds is None:
            raise ValueError('Clip mode set top N_STD but \'num_stds\' parameter not provided')

    dim = 0 if clip == ClipMode.AVG else None
    sat_fn = _get_saturation_fn(mode, clip, num_stds)
    if mode == LinearQuantMode.SYMMETRIC:
        sat_val = sat_fn(tensor, dim)
        scale, zp = symmetric_linear_quantization_params(num_bits, sat_val)
    else:   # Asymmetric mode
        sat_min, sat_max = sat_fn(tensor, dim)
        signed = mode == LinearQuantMode.ASYMMETRIC_SIGNED
        scale, zp = asymmetric_linear_quantization_params(num_bits, sat_min, sat_max, signed=signed)

    return scale, zp


def clamp(input, min, max, inplace=False):
    if inplace:
        return input.clamp_(min, max)
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        return input.mul_(scale).sub_(zero_point).round_()
    return torch.round(scale * input - zero_point)


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        return input.add_(zero_point).div_(scale)
    return (input + zero_point) / scale


def update_ema(biased_ema, value, decay, step):
    biased_ema = biased_ema * decay + (1 - decay) * value
    # FIXME(lizhouyang): pow does not support tensor as exp
    unbiased_ema = biased_ema / (1 - decay ** step.item())  # Bias correction
    return biased_ema, unbiased_ema


class FakeLinearQuantization(nn.Module):
    def __init__(self, num_bits=8, mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, dequantize=True, inplace=False):
        super(FakeLinearQuantization, self).__init__()

        self.num_bits = num_bits
        self.mode = mode
        self.dequantize = dequantize
        self.inplace = inplace
        self.quantizer = qt.AffineQuantizer(1, 0)

        # We track activations ranges with exponential moving average, as proposed by Jacob et al., 2017
        # https://arxiv.org/abs/1712.05877
        # We perform bias correction on the EMA, so we keep both unbiased and biased values and the iterations count
        # For a simple discussion of this see here:
        # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max_biased', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def forward(self, input):
        # We update the tracked stats only in training
        #
        # Due to the way DataParallel works, we perform all updates in-place so the "main" device retains
        # its updates. (see https://pytorch.org/docs/stable/nn.html#dataparallel)
        # However, as it is now, the in-place update of iter_count causes an error when doing
        # back-prop with multiple GPUs, claiming a variable required for gradient calculation has been modified
        # in-place. Not clear why, since it's not used in any calculations that keep a gradient.
        # It works fine with a single GPU. TODO: Debug...

        # skip in quantization simulator
        if hasattr(self, 'is_quant') and self.is_quant:
            return input

        if self.training:
            with torch.no_grad():
                current_min, current_max = get_tensor_min_max(input)
            self.iter_count += 1
            self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data,
                                                                             current_min, self.ema_decay,
                                                                             self.iter_count)
            self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data,
                                                                             current_max, self.ema_decay,
                                                                             self.iter_count)

        if self.mode == LinearQuantMode.SYMMETRIC:
            max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
            actual_min, actual_max = -max_abs, max_abs
            if self.training:
                self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(self.num_bits, max_abs)
        else:
            actual_min, actual_max = self.tracked_min, self.tracked_max
            signed = self.mode == LinearQuantMode.ASYMMETRIC_SIGNED
            if self.training:
                self.scale.data, self.zero_point.data = asymmetric_linear_quantization_params(self.num_bits,
                                                                                              self.tracked_min,
                                                                                              self.tracked_max,
                                                                                              signed=signed)
        self.quantizer = qt.AffineQuantizer(self.scale, self.zero_point)

        input = clamp(input, actual_min.item(), actual_max.item(), False)
        input = F.linear_quantize(input, self.scale, self.zero_point, dtype=input.dtype)
        return input

    def extra_repr(self):
        mode_str = str(self.mode).split('.')[1]
        return 'mode={0}, num_bits={1}, ema_decay={2:.4f})'.format(mode_str, self.num_bits, self.ema_decay)

    def get_output_quantizer(self):
        return self.quantizer


default_params = {
    'bits_weights': 8,
    'bits_acts': 8,
    'algorithm': 'SYMMETRIC',
    'ema_decay': 0.999,
    'with_float_weight': True,
}


class ModuleTransformer(PapeModel):
    """
       伪量化训练，对需要去全低精度训练的层插入Fake Quantization 层模拟
        定点精度损失。

    参数:
        - model(nn.Module): 全精度模型
        - configs(dic): 量化配置，默认如下

    例子:
    >>>  default_params = {
             'bits_weights': 8,
             'bits_acts': 8,
             'algorithm': 'SYMMETRIC',
             'ema_decay': 0.999,
             'with_float_weight': True,
         }
    """

    def __init__(self, model, configs=None):
        super(ModuleTransformer, self).__init__(model)
        if not configs:
            configs = default_params
        if not isinstance(configs['algorithm'], LinearQuantMode):
            configs['algorithm'] = LinearQuantMode[configs['algorithm']]
        self.configs = configs
        self.transform()

    def transform(self):
        self._convert_module(self.model, self.configs)

    def _convert_module_to_quant(self, module, configs):
        # filter the specific unquantized module
        if self._filter_module(module, configs):
            print('skip module:', module.full_name)
            return
        print('convert module:', module.full_name)

        def preforward(self, *inputs):
            # TODO(lizhouyang): enable this for quantize input
            # if hasattr(self, 'input_quant'):
            #     for i in range(len(inputs))
            #         inputs[i] = self.input_quant(inputs[i])
            for param_name, param in self.named_parameters():
                # no fake quant for bias in train
                if param_name.endswith('bias'):
                    continue
                with_fp = param_name.startswith(FP_PREFIX)
                name = param_name[len(FP_PREFIX):] if with_fp else param_name
                with torch.no_grad():
                    scale, zero_point = _get_quant_params_from_tensor(param,
                                                                      configs['bits_weights'],
                                                                      configs['algorithm'])
                setattr(self, name + '_scale', scale)
                setattr(self, name + '_zero_point', zero_point)

                if with_fp:
                    setattr(self, name, F.linear_quantize(param, scale, zero_point, dtype=param.data.dtype))
                else:
                    param.data = F.linear_quantize(param, scale, zero_point, dtype=param.data.dtype)

        def postforward(self, output):
            if hasattr(self, 'output_quant') and self.output_quant is not None:
                output = self.output_quant(output)
            return output

        names = []
        for param_name, param in module.named_parameters():
            # no fake quant for bias in train
            if param_name.endswith('bias'):
                continue
            module.register_buffer(param_name + '_scale', torch.ones(1))
            module.register_buffer(param_name + '_zero_point', torch.zeros(1))
            names.append(param_name)

        if configs['with_float_weight']:
            """
            with_float_weight: If true, will modify layers with weights to keep both a quantized and
                floating-point copy, such that the following flow occurs in each training iteration:
                1. q_weights = quantize(fp_weights)
                2. Forward through network using q_weights
                3. In back-prop:
                    3.1 Gradients calculated with respect to q_weights
                    3.2 We also back-prop through the 'quantize' operation from step 1
                4. Update fp_weights with gradients calculated in step 3.2
            """
            for name in names:
                param = getattr(module, name)
                delattr(module, name)
                module.register_parameter(FP_PREFIX + name, nn.Parameter(param))
                module.register_buffer(name, param.clone())

        # TODO(lizhouyang): enable this for quantize input
        # module.add_module('input_quant',
        #     FakeLinearQuantization(self.num_bits, self.quant_mode, self.ema_decay,
        #                            dequantize=True, inplace=False))
        module.output_quant = FakeLinearQuantization(
            configs['bits_acts'], configs['algorithm'],
            configs['ema_decay'], dequantize=True,
            inplace=getattr(module, 'inplace', False))
        module.preforward = MethodType(preforward, module)
        module.postforward = MethodType(postforward, module)

    def _filter_module(self, model, configs):
        if ('module_filter' in configs and
                isinstance(model, configs['module_filter'])):
            return True
        elif ('module_name_filter' in configs
              and model.full_name in configs['module_name_filter']):
            return True
        return False

    def merge_BN(self):
        for module in self.model.modules():
            if isinstance(module, Conv_BN):
                gamma, beta, mean, var = module.bn.weight.data, \
                                         module.bn.bias.data, \
                                         module.bn.running_mean.data, \
                                         module.bn.running_var.data
                w = module.conv.weight.data
                if module.conv.bias is None:
                    b = torch.zeros_like(beta)
                    module.conv.bias = nn.Parameter(b)
                else:
                    b = module.conv.bias.data
                temp = gamma / (var.sqrt() + 1e-9)
                module.conv.weight.data = temp.reshape(-1, 1, 1, 1) * w
                module.conv.bias.data = temp * (b - mean) + beta
                module.merge_bn = True
                del module.bn

    def _convert_module(self, model, configs):
        for module in model.children():
            if isinstance(module, nn.Sequential):
                for sub in module:
                    self._convert_module(sub, configs)
            else:
                self._convert_module(module, configs)
        else:
            if len(list(model.children())) == 0:
                self._convert_module_to_quant(model, configs)
