import torch
import torch.nn as nn
# from utils.dist_util import HalfOptimizer
from pape.half import HalfModel, HalfOptimizer


def build_optimizer(cfg_trainer, model, sparse=False):
    if cfg_trainer.optimizer.type == 'LARS':
        if not sparse and cfg_trainer.optimizer.kwargs.get('sparse', False):
            cfg_trainer.optimizer.kwargs.sparse = False
        # filter params in bn, bias
        skip_list = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                skip_list.append(name+'.')

        params_skip, params_noskip = [], []
        for name, param in model.named_parameters():
            if any(skip_name in name for skip_name in skip_list):
                params_skip.append(param)
            else:
                params_noskip.append(param)
        param_lrs = [{'params': params_skip, 'skip': True}, {'params': params_noskip}]
    else:
        param_lrs = model.parameters()

    optimizer = getattr(torch.optim, cfg_trainer.optimizer.type)(
                param_lrs, **cfg_trainer.optimizer.kwargs)

    if cfg_trainer.get('mixed_training', None):
        mix_cfg = cfg_trainer.mixed_training
        if mix_cfg.get('half', False) is True:
            optimizer = HalfOptimizer(optimizer, loss_scale=mix_cfg.loss_scale,
                                sparse=sparse)

    return optimizer
