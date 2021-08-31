import torch
import torch.nn as nn
from .. import op
from .. import distributed as dist


def find_module(root, name):
    assert isinstance(name, str)
    module = root
    for module_name in name.split("."):
        module = getattr(module, module_name)
    return module


def is_torchbn_module_name(root, name):
    module = find_module(root, name)
    return is_torchbn_module(module)


def is_syncbn_module_name(root, name):
    module = find_module(root, name)
    return is_syncbn_module(module)


def is_bn_module_name(root, name):
    module = find_module(root, name)
    return is_bn_module(module)


def is_torchbn_module(module):
    return isinstance(module, nn.BatchNorm2d)


def is_syncbn_module(module):
    return isinstance(module, op.SyncBatchNorm2d)


def is_bn_module(module):
    return is_torchbn_module(module) or is_syncbn_module(module)


def convert_syncbn(module, group=dist.group.WORLD):
    """
    convert module's all torch.nn.BatchNorm2d to pape.op.SyncBatchNorm2d.

    Args:
        group (group, optional): Group will be used in SyncBatchNorm2d. (default is group.WORLD)

    Returns:
        module which all BN has been replaced.
    """
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        group_kw = {"group": group}
        module_output = op.SyncBatchNorm2d(module.num_features,
                                           module.eps, module.momentum,
                                           module.affine,
                                           module.track_running_stats,
                                           **group_kw)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        if isinstance(module.num_batches_tracked, int):
            module_output.num_batches_tracked = torch.Tensor([module.num_batches_tracked])
        else:
            module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_syncbn(child, group))
    del module
    return module_output
