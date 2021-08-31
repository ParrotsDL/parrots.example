import torch
from torch.autograd import Variable
from collections import OrderedDict
import copy
from contextlib import contextmanager
from .. import distributed as dist
from ..utils.base import PapeModel


class DistributedParrotsModel(PapeModel):
    """
    DistributedModel to auto average gradients in all models used in Parrots.

    Argument
        * model (Module): Model to distributed. It can be a Module of PyTorch, or a PapeModel
          build with pape
        * require_backward_overlap (bool, optional): if True, will overlap the compute and
          communication in backword. This will accelerate training speed. (default is True)
        * bucket_cap_mb (int, optional): The bucket size (MB) of the communication bucket. (default
          is 1)
        * process_group (group, optional): The group to distributed. (default is ``group.WORLD``)
    """

    def __init__(self, model, require_backward_overlap=True, bucket_cap_mb=1,
                 process_group=dist.group.WORLD):
        super(DistributedParrotsModel, self).__init__(model)

        self.comm_group = process_group
        self.world_size = dist.get_world_size(self.comm_group)
        self.require_backward_grad_sync = self.world_size > 1
        self.require_backward_overlap = require_backward_overlap
        self.bucket_cap = bucket_cap_mb * 1024 * 1024
        self.param_type_dict = OrderedDict({"Float16": 0,
                                            "Float32": 1,
                                            "Float64": 2})
        self.broadcast_parameters()

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations.
        Within this context, gradients will be accumulated on module
        variables, which will later be synchronized when the first
        forward-backward pass exits the context.
        Example::
            >>> dm = pape.parallel.DistributedModel(model)
            >>> with dm.no_sync():
            ...   for input in inputs:
            ...     dm(input).backward()      # accumulate grads
            ...     dm.average_gradients()    # no synchronize grads
            ... dm(another_input).backward()
            ... dm.average_gradients()        # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    def broadcast_parameters(self):
        if self.world_size > 1:
            for _, param in self.model.state_dict().items():
                dist.broadcast(param.data, 0, group=self.comm_group)

    def average_gradients(self):
        """
        Do the average gradients after ``backward()`` in Parrots.
        """
        if not self.require_backward_grad_sync:
            return

        buckets = [[] for _ in self.param_type_dict.keys()]
        if self.require_backward_overlap:
            # actually, parrots cannot overlap in 0.4, will resolve this in 0.5
            buckets_size = [0 for _ in self.param_type_dict.keys()]
            param_reversed_list = reversed(list(self.model.parameters()))
            for param in param_reversed_list:
                if param.requires_grad:
                    bucket_idx = self.param_type_dict[param.spec.dtype.name]

                    buckets[bucket_idx].append(param.grad)
                    buckets_size[bucket_idx] += param.grad.numel() * param.element_size()
                    if buckets_size[bucket_idx] >= self.bucket_cap:
                        dist.all_reduce(buckets[bucket_idx], group=self.comm_group)
                        buckets[bucket_idx].clear()
                        buckets_size[bucket_idx] = 0
        else:
            for param in self.model.parameters():
                if param.requires_grad:
                    bucket_idx = self.param_type_dict[param.spec.dtype.name]
                    buckets[bucket_idx].append(param.grad)

        for bucket in buckets:
            if len(bucket) > 0:
                dist.all_reduce(bucket, group=self.comm_group)

        for param in self.model.parameters():
            if param.requires_grad:
                param.grad.data /= self.world_size


def flatten(bucket):
    return torch._utils._flatten_dense_tensors(bucket)


def unflatten(coalesced, bucket):
    return torch._utils._unflatten_dense_tensors(coalesced, bucket)


class DistributedPyTorchModel(PapeModel):
    """
    DistributedModel to auto average the gradients in all models used in PyTorch.

    Argument
        * model (Module): Model to distributed. It can be a Module of PyTorch, or a PapeModel
          build with pape
        * require_backward_overlap (bool, optional): if True, will overlap the compute and
          communication in backword. This will accelerate the training speed. (default is True)
        * bucket_cap_mb (int, optional): The bucket size (MB) of the communication bucket. (default
          is 1)
        * process_group (group, optional): The group to distributed. (default is ``group.WORLD``)
    """

    def __init__(self, model, require_backward_overlap=True, bucket_cap_mb=1,
                 process_group=dist.group.WORLD):
        super(DistributedPyTorchModel, self).__init__(model)

        self.comm_group = process_group
        self.world_size = dist.get_world_size(self.comm_group)
        self.require_backward_grad_sync = self.world_size > 1
        self.require_backward_overlap = require_backward_overlap
        self.bucket_cap = bucket_cap_mb * 1024 * 1024

        self.current_stream = None
        self.overlap_stream = torch.cuda.Stream(priority=0)
        self.event = torch.cuda.Event(enable_timing=False, blocking=False)

        self.param_type_dict = OrderedDict({"torch.cuda.HalfTensor": 0,
                                            "torch.cuda.FloatTensor": 1,
                                            "torch.cuda.DoubleTensor": 2})
        self.create_hooks()
        self.broadcast_model()

    def __setstate__(self, state):
        super(DistributedPyTorchModel, self).__setstate__(state)

        self.current_stream = None
        self.overlap_stream = torch.cuda.Stream(priority=0)
        self.event = torch.cuda.Event(enable_timing=False, blocking=False)

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        del attrs['self.current_stream']
        del attrs['self.overlap_stream']
        del attrs['self.event']
        return attrs

    def split_params(self, tensors):
        dtypes = self.param_type_dict.keys()
        buckets = []
        for i, dtype in enumerate(dtypes):
            bucket = [t for t in tensors if t.type() == dtype]
            if bucket:
                buckets.append(bucket)
        return buckets

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations.
        Within this context, gradients will be accumulated on module
        variables, which will later be synchronized when the first
        forward-backward pass exits the context.
        Example::
            >>> dm = pape.parallel.DistributedModel(model)
            >>> with dm.no_sync():
            ...   for input in inputs:
            ...     dm(input).backward()      # no synchronization, accumulate grads
            ... dm(another_input).backward()  # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    def broadcast_model(self):
        if self.world_size > 1:
            for _, param in self.model.state_dict().items():
                dist.broadcast(param.data, 0, group=self.comm_group)

    def create_hooks(self):

        def wait_backward_overlap_done():
            for bucket in self.buckets:
                if len(bucket) > 0:
                    self.allreduce_bucket(bucket, bucket_stream=self.overlap_stream)
            self.overlap_stream.record_event(self.event)
            self.current_stream.wait_event(self.event)

        self.grad_accs = []
        for param in self.parameters():
            if param.requires_grad:
                def wrapper(param):
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]

                    def allreduce_hook(*unused):
                        if self.require_backward_grad_sync:
                            if self.require_backward_overlap:
                                if not self.callback_queued:
                                    Variable._execution_engine.queue_callback(wait_backward_overlap_done)
                                    self.callback_queued = True
                                self.try_allreduce_buckets(param)
                            else:
                                if not self.callback_queued:
                                    Variable._execution_engine.queue_callback(self.allreduce_all_params)
                                    self.callback_queued = True

                    grad_acc.register_hook(allreduce_hook)
                    self.grad_accs.append(grad_acc)

                wrapper(param)

    def allreduce_bucket(self, bucket, bucket_stream):
        self.current_stream.record_event(self.event)
        bucket_stream.wait_event(self.event)

        with torch.cuda.stream(bucket_stream):
            tensor = flatten(bucket)
            dist.all_reduce(tensor, group=self.comm_group)
            tensor.div_(self.world_size)
            for buf, synced in zip(bucket, unflatten(tensor, bucket)):
                buf.copy_(synced)

    def allreduce_all_params(self):
        grads = [param.grad.data for param in self.parameters() if param.grad is not None]
        split_buckets = self.split_params(grads)

        for _, bucket in enumerate(split_buckets):
            self.allreduce_bucket(bucket, bucket_stream=self.current_stream)

    def try_allreduce_buckets(self, param):
        bucket_idx = self.param_type_dict[param.type()]

        self.buckets[bucket_idx].append(param.grad.data)
        self.buckets_size[bucket_idx] += param.grad.data.numel() * param.element_size()

        if self.buckets_size[bucket_idx] >= self.bucket_cap:
            self.allreduce_bucket(self.buckets[bucket_idx], bucket_stream=self.overlap_stream)
            self.buckets[bucket_idx] = []
            self.buckets_size[bucket_idx] = 0

    def forward(self, *inputs, **kwargs):
        result = self.model(*inputs, **kwargs)

        if self.require_backward_grad_sync:
            self.current_stream = torch.cuda.current_stream()
            if self.require_backward_overlap:
                self.buckets = [[] for _ in self.param_type_dict.keys()]
                self.buckets_size = [0 for _ in self.param_type_dict.keys()]
            self.callback_queued = False

        return result

    def average_gradients(self):
        """
        Don't need use in pytorch
        """
        pass
