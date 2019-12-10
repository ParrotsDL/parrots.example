import torch
from pape.half import HalfOptimizer


def build_optimizer(cfg_trainer, model):
    if cfg_trainer.get('mixed_training', False):
        optimizer = HalfOptimizer(model, cfg_trainer.optimizer.type, **cfg_trainer.optimizer.kwargs)
    else:
        optimizer = getattr(torch.optim, cfg_trainer.optimizer.type)(
                    model.parameters(), **cfg_trainer.optimizer.kwargs)
    return optimizer
