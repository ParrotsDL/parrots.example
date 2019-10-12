import torch
try:
    from pape.parallel import DistributedModel
    from pape.half import HalfModel, HalfOptimizer
    from pape.distributed import init, barrier, get_rank, get_world_size, all_reduce, group
    from pape.op import SyncBatchNorm2d
except ImportError:
    '''
    This is a fake pape
    '''
    import os
    import torch.distributed as dist
    from torch.nn import Module

    class DistributedModel(Module):
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
                if param.requires_grad:
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

    class HalfModel(Module):
        def __init__(self, module, float_layers=None):
            raise NotImplementedError(
                "Mix training mode not supported in NON-PAPE environment.")

    class HalfOptimizer(object):
        def __init__(self, half_model, optimizer_name, **kwargs):
            raise NotImplementedError(
                "Mix training optimizer not supported in NON-PAPE environment.")

    class group(object):
        WORLD = 0

    class SyncBatchNorm2d(Module):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                     track_running_stats=True, group=dist.group.WORLD):
            raise NotImplementedError(
                "SyncBN not supported in NON-PAPE environment.")

    def get_rank():
        return int(os.environ.get('SLURM_PROCID', 0))

    def get_world_size():
        return int(os.environ.get('SLURM_NTASKS', 1))

    def init():
        node_list = os.environ['SLURM_NODELIST']
        if '[' in node_list:
            beg = node_list.find('[')
            pos1 = node_list.find('-', beg)
            if pos1 < 0:
                pos1 = 1000
            pos2 = node_list.find(',', beg)
            if pos2 < 0:
                pos2 = 1000
            node_list = node_list[:min(pos1, pos2)].replace('[', '')
        hostname = node_list[8:].replace('-', '.')
        global_rank = get_rank()
        global_size = get_world_size()
        num_gpus = torch.cuda.device_count()
        local_rank = global_rank % num_gpus
        torch.cuda.set_device(local_rank)

        os.environ['MASTER_PORT'] = str(12345)
        os.environ['MASTER_ADDR'] = hostname
        os.environ['WORLD_SIZE'] = str(global_size)
        os.environ['RANK'] = str(global_rank)
        if global_size > 1:
            dist.init_process_group(backend='nccl')
        return global_rank, global_size, local_rank

    def barrier():
        if get_world_size() > 1:
            x = torch.cuda.IntTensor([1])
            dist.all_reduce(x)
            x.cpu().numpy()

    def all_reduce(tensor, op=dist.reduce_op.SUM, group=dist.group.WORLD):
        dist.all_reduce(tensor, op, group)
