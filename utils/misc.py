import numpy as np
import torch
import logging
logger = logging.getLogger()


def check_keys(model, checkpoint):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint['model'].keys())
    missing_keys = model_keys - ckpt_keys
    for key in missing_keys:
        logger.warning('missing key in model:{}'.format(key))
    unexpected_keys = ckpt_keys - model_keys
    for key in unexpected_keys:
        logger.warning('unexpected key in checkpoint:{}'.format(key))
    shared_keys = model_keys & ckpt_keys
    return shared_keys


class AverageMeter(object):
    """Computes and stores the average and current value
       When length < 0 , save all history data """

    def __init__(self, name, fmt=':f', length=1):
        self.name = name
        self.fmt = fmt
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 1:
            self.history = []
        elif self.length < 0:
            self.count = 0
            self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val):
        self.val = val
        if self.length > 1:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.avg = np.mean(self.history)
        elif self.length < 0:
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

    def __str__(self):
        if self.length == 0 or self.length == 1:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
