import torch
import torch.nn as nn


def get_layers(model, layers):
    for layer in model.children():
        if list(layer.children()) == []:
            layers.append(layer)
        else:
            get_layers(layer, layers)


def get_params(model):
    layers = []
    get_layers(model, layers)
    params_bn, params_other = [], []
    for i, layer in enumerate(layers):
        layer_dict = dict(layer._parameters.items())
        if type(layer) == nn.BatchNorm2d:
            if 'weight' in layer_dict:
                params_bn.append(layer.weight)
            if 'bias' in layer_dict:
                if layer_dict['bias'] is not None:
                    params_bn.append(layer.bias)
        else:
            if 'weight' in layer_dict:
                params_other.append(layer.weight)
            if 'bias' in layer_dict:
                if layer_dict['bias'] is not None:
                    params_other.append(layer.bias)
    return params_bn, params_other


def build_optimizer(model, cfgs):
    optimizer = torch.optim.SGD(model.parameters(), **cfgs.trainer.optimizer.kwargs)
    if cfgs.trainer.get('bn', None):
        if cfgs.trainer.bn.get('weight_decay', True) is False:
            params_bn, params_other = get_params(model)
            optimizer = torch.optim.SGD([
                                         {'params': params_bn, 'weight_decay': 0.0},
                                         {'params': params_other}
                                        ], **cfgs.trainer.optimizer.kwargs)
    return optimizer
