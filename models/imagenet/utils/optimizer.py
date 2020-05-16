import torch
import torch.nn as nn
# from utils.dist_util import HalfOptimizer
from pape.half import HalfOptimizer


def build_optimizer(cfg_trainer, model, sparse=False):
    if cfg_trainer.optimizer.type == 'LARS':
        if not sparse and cfg_trainer.optimizer.kwargs.get('sparse', False):
            cfg_trainer.optimizer.kwargs.sparse = False
        # filter params in bn, bias
        skip_list = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                skip_list.append(name+'.')

        param_lrs = []
        param_reversed_list = reversed(list(model.named_parameters()))
        for name, param in param_reversed_list:
            if any(skip_name in name for skip_name in skip_list):
                param_lrs.append({'params': [param], 'skip': True, 'weight_decay': 0.0})
            elif 'bias' in name:
                param_lrs.append({'params': [param], 'weight_decay': 0.0})
            else:
                param_lrs.append({'params': [param]})
    else:
        # filter params in bn, bias
        skip_list = ['bias']
        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                skip_list.append(name+'.')

        param_lrs = []
        param_reversed_list = reversed(list(model.named_parameters()))
        for name, param in param_reversed_list:
            if any(skip_name in name for skip_name in skip_list):
                param_lrs.append({'params': [param], 'weight_decay': 0.0})
            else:
                param_lrs.append({'params': [param]})
        # param_lrs = []
        # for module_ in model.modules():
        #     weight_decay_params = {'params': []}
        #     no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
        #     if isinstance(module_, nn.BatchNorm2d):
        #         no_weight_decay_params['params'].extend(
        #             [p for p in list(module_._parameters.values())
        #              if p is not None])
        #     else:
        #         weight_decay_params['params'].extend(
        #             [p for n, p in list(module_._parameters.items())
        #              if p is not None and n != 'bias'])
        #         no_weight_decay_params['params'].extend(
        #             [p for n, p in list(module_._parameters.items())
        #              if p is not None and n == 'bias'])
        #     if weight_decay_params['params']:
        #         param_lrs.append(weight_decay_params)
        #     if no_weight_decay_params['params']:
        #         param_lrs.append(no_weight_decay_params)

    optimizer = getattr(torch.optim, cfg_trainer.optimizer.type)(
                param_lrs, **cfg_trainer.optimizer.kwargs)

    if cfg_trainer.get('mixed_training', None):
        mix_cfg = cfg_trainer.mixed_training
        if mix_cfg.get('half', False) is True:
            optimizer = HalfOptimizer(optimizer, loss_scale=mix_cfg.loss_scale, sparse=sparse)

    return optimizer
