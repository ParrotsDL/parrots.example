from torch.optim import Optimizer
from utils.dist_util import HalfOptimizer
import math
from addict import Dict


def get_scheduler(optimizer, data_size, max_epoch, cfg):
    cfg = Dict(cfg)
    if cfg.type == 'MultiStepLR':
        return IterLRScheduler(optimizer, data_size, cfg.kwargs.milestones,
                               cfg.kwargs.gamma)
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
        # self.step(last_iter + 1)
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
