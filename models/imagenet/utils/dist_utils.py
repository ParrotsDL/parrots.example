import torch.distributed as dist
import torch.nn as nn


class DistributedModel(nn.Module):
    def __init__(self, model):
        super(DistributedModel, self).__init__()
        self.model = model
        self.broadcast_params()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistributedModel, self).train(mode)
        self.model.train(mode)

    def average_gradients(self):
        world_size = dist.get_world_size()
        param_list = []
        for param in self.model.parameters():
            # if param.requires_grad:
            if param.requires_grad and param.grad.data is not None:
                dist.all_reduce(param.grad.data)
                param_list.append(param)
        for param in param_list:
            param.grad.data /= world_size

    def sum_gradients(self):
        for param in self.model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data)

    def broadcast_params(self):
        for p in self.model.state_dict().values():
            dist.broadcast(p, 0)
