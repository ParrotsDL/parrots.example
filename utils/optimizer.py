import torch
from utils.dist_util import HalfOptimizer


def build_optimizer(cfg_trainer, model):
    if cfg_trainer.get('mixed_precision', None):
        mix_cfg = cfg_trainer.mixed_precision
        if mix_cfg.get('half', False) is True:
            optimizer = torch.optim.SGD(model.parameters(), **cfg_trainer.optimizer.kwargs)
            optimizer = HalfOptimizer(optimizer, loss_scale=cfg_trainer.mixed_precision.loss_scale)
        else:
            optimizer = getattr(torch.optim, cfg_trainer.optimizer.type)(
                        model.parameters(), **cfg_trainer.optimizer.kwargs)
    else:
        optimizer = getattr(torch.optim, cfg_trainer.optimizer.type)(
                    model.parameters(), **cfg_trainer.optimizer.kwargs)
    return optimizer
