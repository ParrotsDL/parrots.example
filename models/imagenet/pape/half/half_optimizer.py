import logging
import torch
from ..utils.base import PapeOptimizer

logger = logging.getLogger()


class HalfOptimizer(PapeOptimizer):
    """
    Half optimizer for HalfModel training.

    The idea of dynamic loss scaling comes from [Apex](https://nvidia.github.io/apex/index.html).
    Dynamic loss scaling first attempts very high loss scale (``dynamic_scale_init``).
    Ironically, it may result in OVERflowing gradients.
    If overflowing gradients are encountered, ``HalfOptimizer`` then skips the update step for
    this particular iteration/minibatch and decreases the loss scale (divide by ``dynamic_scale_factor``).
    If overflowing gradients haven't occured after a certain number of iterations (``dynamic_scale_window``),
    then increases the loss scale (multiply ``dynamic_scale_factor``).

    In this way, ``HalfOptimizer`` attempts to "ride the edge" by
    always using the highest possible loss scale without incurring overflow.

    Arguments:
        - optimizer (Optimizer): The optimizer to be changed to half optimizer.
        - loss_scale (float or ``'dynamic'``, optional): If float, will use static loss scale; if string 'dynamic',
          will use dynamic loss scale. (default is 'dynamic')
        - dynamic_scale_init (float, optional): Init value of dynamic loss scale. (default is 2**32)
        - dynamic_scale_factor (float, optional): Factor to change loss scale. (default is 2.0)
        - dynamic_scale_window (int, optional): Number of iters without overflow before increasing
          dynamic loss scale. (default is 1000)
    """

    def __init__(self, optimizer, loss_scale='dynamic', dynamic_scale_init=2**32,
                 dynamic_scale_factor=2., dynamic_scale_window=1000):
        super(HalfOptimizer, self).__init__(optimizer)

        if isinstance(loss_scale, str) and loss_scale == 'dynamic':
            self.dynamic = True
            self.loss_scale = dynamic_scale_init
            self.dynamic_scale_factor = dynamic_scale_factor
            self.dynamic_scale_window = dynamic_scale_window
            self.non_overflow_iter = 0
        elif isinstance(loss_scale, (float, int)):
            self.dynamic = False
            self.loss_scale = loss_scale
        else:
            assert False, "loss_scale must be number or string 'dynamic', but get {}".format(loss_scale)

        self.model_param_groups = self.optimizer.param_groups
        self.master_param_groups = []
        for param_group in self.model_param_groups:
            master_param_group = {}
            for key, value in param_group.items():
                if key == 'params':
                    float_params = []
                    for param in value:
                        if isinstance(param, (torch.cuda.HalfTensor, torch.HalfTensor)):
                            float_param = param.clone().detach().float()
                            float_param.requires_grad = True
                        else:
                            float_param = param
                        float_params.append(float_param)
                    master_param_group['params'] = float_params
                else:
                    master_param_group[key] = value
            self.master_param_groups.append(master_param_group)
        self.optimizer.param_groups = self.master_param_groups

    def zero_grad(self):
        for param_group in self.model_param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        self.optimizer.zero_grad()

    def step(self, closure=None):
        for model_group, master_group in zip(self.model_param_groups, self.master_param_groups):
            model_params = model_group['params']
            master_params = master_group['params']
            for model_param, master_param in zip(model_params, master_params):
                if model_param.grad is not None:
                    if isinstance(model_param.grad, (torch.cuda.HalfTensor, torch.HalfTensor)):
                        master_param.grad = model_param.grad.clone().detach().float()
                else:
                    master_param.grad = None

                if master_param.grad is not None:
                    master_param.grad.div_(self.loss_scale)
                    if self.dynamic:
                        param_sum = master_param.grad.sum()
                        # in parrots, has bug torch.isfinite(torch.tensor([float('nan')])) is True
                        # if not torch.isfinite(param_sum).item():
                        if torch.isinf(param_sum).item() or torch.isnan(param_sum).item():
                            self.non_overflow_iter = 0
                            old_loss_scale = self.loss_scale
                            self.loss_scale /= self.dynamic_scale_factor
                            logger.info("Gradients overflow! Decrease loss scale from {} to {}".format(
                                        old_loss_scale, self.loss_scale))
                            return

        if self.dynamic:
            self.non_overflow_iter += 1
            if self.non_overflow_iter >= self.dynamic_scale_window:
                self.non_overflow_iter = 0
                old_loss_scale = self.loss_scale
                self.loss_scale *= self.dynamic_scale_factor
                logger.info("Gradients not overflow in {} iters. Increase loss scale from {} to {}".format(
                            self.dynamic_scale_window, old_loss_scale, self.loss_scale))

        res = self.optimizer.step(closure)

        for model_group, master_group in zip(self.model_param_groups, self.master_param_groups):
            model_params = model_group['params']
            master_params = master_group['params']
            for model_param, master_param in zip(model_params, master_params):
                if isinstance(model_param.data, (torch.cuda.HalfTensor, torch.HalfTensor)):
                    model_param.data.copy_(master_param.data)

        return res

    def scale_up_loss(self, loss):
        """
        Multiply the loss_scale with loss. This function must be called before ``backward()``

        Arguments:
            * loss (Loss): loss to be multiply

        Return:
            A new loss ``loss * self.loss_scale``
        """
        return loss * self.loss_scale

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['dynamic'] = self.dynamic
        state_dict['loss_scale'] = self.loss_scale
        if self.dynamic:
            state_dict['non_overflow_iter'] = self.non_overflow_iter
            state_dict['dynamic_scale_factor'] = self.dynamic_scale_factor
            state_dict['dynamic_scale_window'] = self.dynamic_scale_window
        return state_dict

    def load_state_dict(self, state_dict):
        self.dynamic = state_dict.pop('dynamic')
        self.loss_scale = state_dict.pop('loss_scale')
        if self.dynamic:
            self.non_overflow_iter = state_dict.pop('non_overflow_iter')
            self.dynamic_scale_factor = state_dict.pop('dynamic_scale_factor')
            self.dynamic_scale_window = state_dict.pop('dynamic_scale_window')
        self.optimizer.load_state_dict(state_dict)

    def __getstate__(self):
        state = self.optimizer.__getstate__()
        state['dynamic'] = self.dynamic
        state['loss_scale'] = self.loss_scale
        if self.dynamic:
            state['non_overflow_iter'] = self.non_overflow_iter
            state['dynamic_scale_factor'] = self.dynamic_scale_factor
            state['dynamic_scale_window'] = self.dynamic_scale_window
        return state

    def __setstate(self, state):
        self.dynamic = state['dynamic']
        self.loss_scale = state['loss_scale']
        if self.dynamic:
            self.non_overflow_iter = state['non_overflow_iter']
            self.dynamic_scale_factor = state['dynamic_scale_factor']
            self.dynamic_scale_window = state['dynamic_scale_window']
        self.optimizer.__setstate__(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' (\n'
        format_string += 'dynamic={dynamic}, loss_scale={loss_scale}'.format(
                         **self.__dict__)
        format_string += '\n' + repr(self.optimizer) + '\n)'
        return format_string

    def add_param_group(self, param_group):
        self.optimizer.param_groups = self.model_param_groups
        self.optimizer.add_param_group(param_group)
        self.model_param_groups = self.optimizer.param_groups

        last_model_param_group = self.model_param_groups[-1]
        last_master_param_group = {}
        for key, value in last_model_param_group.items():
            if key == 'params':
                float_params = []
                for param in value:
                    if isinstance(param, (torch.cuda.HalfTensor, torch.HalfTensor)):
                        float_param = param.clone().detach().float()
                    else:
                        float_param = param
                    float_params.append(float_param)
                last_master_param_group['params'] = float_params
            else:
                last_master_param_group[key] = value
        self.master_param_groups.append(last_master_param_group)

        self.optimizer.param_groups = self.master_param_groups
