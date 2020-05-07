from torch.optim import Optimizer
from pape.half import HalfModel, HalfOptimizer
import math
from addict import Dict


def get_scheduler(optimizer, data_size, max_epoch, max_iter, cfg):
    cfg = Dict(cfg)
    if cfg.type == 'MultiStepLR':
        milestones = [mile * data_size for mile in cfg.kwargs.milestones]
        return MultiStepLRScheduler(optimizer, milestones, cfg.kwargs.gamma,
                                    cfg.base_lr, cfg.warmup_lr,
                                    cfg.warmup_epochs*data_size)
    elif cfg.type == 'PolyLR':
        return PolyLRScheduler(optimizer, cfg.base_lr, cfg.warmup_lr,
                               cfg.warmup_epochs*data_size, cfg.kwargs.power, max_iter)
    elif cfg.type == 'EpochStepLR':
        milestones = range(cfg.kwargs.perXepochs, max_epoch + 1,
                           cfg.kwargs.perXepochs)
        return IterLRScheduler(optimizer, data_size, milestones,
                               cfg.kwargs.gamma)
    elif cfg.type == 'LinearLR':
        return LinearLRScheduler(optimizer, data_size * max_epoch,
                                 cfg.kwargs.min_lr)
    elif cfg.type == 'CosineLR':
        return CosineLRScheduler(optimizer, data_size * max_epoch,
                                 cfg.kwargs.min_lr)
    else:
        raise Exception('unknown lr scheduler type: {}'.format(cfg.type))


class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not (isinstance(optimizer, (Optimizer, HalfOptimizer))):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".
                        format(i))

        self.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(
            map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        for param_group, lr in zip(self.optimizer.param_groups,
                                   self._get_new_lr()):
            param_group['lr'] = lr


class _WarmUpLRScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 base_lr,
                 warmup_lr,
                 warmup_steps,
                 last_iter=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        if warmup_steps == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            # first compute relative scale for self.base_lr, then multiply to base_lr
            scale = (
                (self.last_iter / self.warmup_steps) *
                (self.warmup_lr - self.base_lr) + self.base_lr) / self.base_lr
            #print('last_iter: {}, warmup_lr: {}, base_lr: {}, scale: {}'.format(self.last_iter, self.warmup_lr, self.base_lr, scale))
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None


class MultiStepLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, milestones, gamma, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        super(MultiStepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.milestones = milestones
        if isinstance(gamma, list):
            assert len(milestones) == len(gamma)
            self.lr_mults = gamma
        else:
            self.lr_mults = [gamma] * len(milestones)
        self.base_lrs = [self.base_lr for group in optimizer.param_groups]

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        try:
            pos = self.milestones.index(self.last_iter)
        except ValueError:
            return list(
                map(lambda group: group['lr'], self.optimizer.param_groups))
        except Exception:
            raise Exception(
                'Unknown Exception occured while getting learning rate...')
        return list(
            map(lambda group: group['lr'] * self.lr_mults[pos],
                self.optimizer.param_groups))


class PolyLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, power, max_iter, last_iter=-1):
        super(PolyLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_steps, last_iter)
        self.power = power
        self.max_iter = max_iter
        self.base_lrs = [self.base_lr for group in optimizer.param_groups]

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        lr_ratio = (1 - float(self.last_iter) / self.max_iter) ** self.power
        return [
            self.warmup_lr * lr_ratio for group in self.optimizer.param_groups
        ]


class IterLRScheduler(_LRScheduler):
    def __init__(self, optimizer, data_size, milestones, gamma, last_iter=-1):
        super(IterLRScheduler, self).__init__(optimizer, last_iter)
        self.milestones = [mile * data_size for mile in milestones]
        if isinstance(gamma, list):
            assert len(milestones) == len(gamma)
            self.lr_mults = gamma
        else:
            self.lr_mults = [gamma] * len(milestones)

    def _get_new_lr(self):
        try:
            pos = self.milestones.index(self.last_iter)
        except ValueError:
            return list(
                map(lambda group: group['lr'], self.optimizer.param_groups))
        except Exception:
            raise Exception(
                'Unknown Exception occured while getting learning rate...')
        return list(
            map(lambda group: group['lr'] * self.lr_mults[pos],
                self.optimizer.param_groups))


class CosineLRScheduler(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_iter=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineLRScheduler, self).__init__(optimizer, last_iter)

    def _get_new_lr(self):
        cosine_ratio = (
            1 + math.cos(math.pi * self.last_iter / self.T_max)) / 2
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_ratio
            for base_lr in self.base_lrs
        ]


class LinearLRScheduler(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_iter=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(LinearLRScheduler, self).__init__(optimizer, last_iter)

    def _get_new_lr(self):
        linear_ratio = 1 - self.last_iter / self.T_max
        return [
            self.eta_min + (base_lr - self.eta_min) * linear_ratio
            for base_lr in self.base_lrs
        ]
