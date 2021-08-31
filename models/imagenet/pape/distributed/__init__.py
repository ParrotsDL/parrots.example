import logging
import torch
import atexit
from ..utils import build_helper
pytorch_version, parrots_version = build_helper.get_env_version()
if pytorch_version:
    from .. import distributed_ext as distributed_ext
else:
    from . import parrots_distributed_ext as distributed_ext

logger = logging.getLogger('pape')

_INITIALIZED = 1
_initialized = 0


def init():
    """
    Init the distributed backend. This will initialize the mpi and nccl
    backend and default process group ``group.WORLD``.

    Also, this will set the cuda device with ``torch.cuda.set_device(local_rank)``.

    .. note::
        If you set cuda device manually in parrots, it will cause an error. But you can set cuda
        device manually in pytorch.
    """
    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize pape.distributed twice!")

    distributed_ext.initialize()

    _initialized = _INITIALIZED
    atexit.register(distributed_ext.finalize)

    try:
        import parrots # noqa
    except ImportError:
        import torch
        torch.cuda.set_device(get_local_rank())

    if get_rank() == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    return get_rank(), get_world_size(), get_local_rank()


def is_initialized():
    """
    Check if the ``distributed.init()`` has been called.
    """
    return _initialized == _INITIALIZED


class ReduceOp(object):
    """
    An enum-like class for available reduction operations: SUM, PRODUCT, MIN, MAX.

    Actually, ReduceOp.SUM = 0, ReduceOp.PRODUCT = 1, ReduceOp.MIN = 2, ReduceOp.MAX = 3.

    As a parameter in pape.distributed, you can use string ``'sum'``, ``'product'``, ``'min'``,
    ``'max'`` instead of ``ReduceOp.SUM``, ``ReduceOp.PRODUCT``, ``ReduceOp.MIN``, ``ReduceOp.MAX``.
    """
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3

    @staticmethod
    def get(name):
        if isinstance(name, str):
            return getattr(ReduceOp, name.upper())
        elif isinstance(name, int):
            return name
        else:
            raise Exception("Parameter not support: {}".format(name))


class group(object):
    """
    An enum-like class provides the default communication group.

    Actually, group.WORLD = 0.
    """
    WORLD = 0


class Backend(object):
    """
    An enum-like class of available backends: AUTO, MPI, NCCL.

    Actually, Backend.AUTO = 0, Backend.MPI = 1, Backend.NCCL = 2.

    As a parameter in pape.distributed, you can use string ``'auto'``, ``'mpi'``, ``'nccl'``
    instead of ``Backend.AUTO``, ``Backend.MPI``, ``Backend.NCCL``.
    """
    AUTO = 0
    MPI = 1
    NCCL = 2

    @staticmethod
    def get(name):
        if isinstance(name, str):
            return getattr(Backend, name.upper())
        elif isinstance(name, int):
            return name
        else:
            raise Exception("Parameter not support: {}".format(name))


def get_rank(group=group.WORLD):
    """
    Returns the rank in current process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Arguments:
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)

    Returns:
        The rank of this process in group
    """
    assert _initialized
    return distributed_ext.get_rank(group)


def get_world_size(group=group.WORLD):
    """
    Returns the number of processes in current process group.

    Arguments:
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)

    Returns:
        The world size of the process group
    """
    assert _initialized
    return distributed_ext.get_world_size(group)


def get_local_rank():
    """
    Returns the local rank in current host.

    Local rank is the process rank in current host. e.g. when setup a job
    with 16 processes in two hosts, their (global) rank is 0 - 15, their local rank
    in host1 is 0 - 7, in host2 is also 0 - 7.

    The local rank is used as the parameter of ``torch.cuda.set_device()``.
    But you don't need call set_device() manually, the ``distributed.init()`` already finish the job.

    Returns:
        The local rank
    """
    assert _initialized
    return distributed_ext.get_local_rank()


def send(tensor, dst, group=group.WORLD, tag=0, backend=Backend.AUTO):
    """
    Sends a tensor to process ``dst``.

    Arguments:
        * tensor (Tensor): Tensor to be send
        * dst (int): Destination rank
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * tag (int, optional): Tag to match ``send`` with remote ``recv``
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    return distributed_ext.send(tensor, dst, group, tag, backend)


def recv(tensor, src=None, group=group.WORLD, tag=0, backend=Backend.AUTO):
    """
    Receives a tensor.

    Arguments:
        * tensor (Tensor): Tensor to be filled with received data
        * src (int, optional): Source rank. Will receive from any
          process if unspecified
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * tag (int, optional): Tag to match ``recv`` with remote ``send``
        * backend (Backend, optional): The backend to use (default is ``'auto'``)

    Returns:
        Sender rank
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    if src is not None:
        distributed_ext.recv(tensor, src, group, tag, backend)
        return src
    else:
        return distributed_ext.recv_all(tensor, group, tag, backend)


def broadcast(tensor, src, group=group.WORLD, backend=Backend.AUTO):
    """
    Broadcasts the tensor to the whole group's processes.

    ``tensor`` must have the same number of elements in all processes
    in the group.

    Arguments:
        * tensor (Tensor): Tensor to be sent if ``src`` is the rank of current
          process or tensor to be used to save received data otherwise
        * src (int): Source rank
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    return distributed_ext.broadcast(tensor, src, group, backend)


def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD,
               backend=Backend.AUTO):
    """
    Reduces the target tensors in all processes to a single tensor and returns
    the resultant tensor to all processes

    After the call, ``tensor`` will be bitwise identical in all processes.

    Arguments:
        * tensor (Tensor): Input and output of the collective. The function
          operates in-place
        * op (ReduceOp, optional): Operation used for element-wise reductions (default is ``'sum'``)
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    op = ReduceOp.get(op)
    return distributed_ext.all_reduce(tensor, op, group, backend)


def reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD,
           backend=Backend.AUTO):
    """
    Reduces the target tensors across all processes to one process.

    Only the process ``dst`` is going to receive the final result.

    Arguments:
        * tensor (Tensor): Input and output of the collective. The function
          operates in-place
        * dst (int): Destination rank
        * op (ReduceOp, optional): Operation used for element-wise reductions (default is ``'sum'``)
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    op = ReduceOp.get(op)
    return distributed_ext.reduce(tensor, dst, op, group, backend)


def all_gather(tensor_send, tensor_recv, group=group.WORLD, backend=Backend.AUTO):
    """
    Gathers tensors from the whole group into all processes.

    Arguments:
        * tensor_send (Tensor): Tensor to be send from current process
        * tensor_recv (Tensor or list[Tensor]): Output can be a tensor or a list of tensors.
          When use a tensor to ``recv`` all data from all processes' tensor_send, its
          size should be ``get_world_size(group) * tensor_send.numel()``. When use a list of tensors,
          the size of the list should be ``get_world_size(group)`` and all the tensors in the list
          should have size ``tensor_send.numel()``.
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    if isinstance(tensor_recv, list):
        world_size = get_world_size(group)
        tmp_recv = torch.empty([tensor_send.numel()*world_size],
                               dtype=tensor_send.dtype,
                               device=tensor_send.device)
        distributed_ext.all_gather(tensor_send, tmp_recv, group, backend)
        tmp_recv = tmp_recv.reshape([world_size, -1])
        for i in range(world_size):
            tensor_recv[i].copy_(tmp_recv[i])
    else:
        return distributed_ext.all_gather(tensor_send, tensor_recv, group, backend)


def reduce_scatter(tensor_send, tensor_recv, op=ReduceOp.SUM, group=group.WORLD,
                   backend=Backend.AUTO):
    """
    Reduces tensors from all processes, then scatter the result to all processes.

    Arguments:
        * tensor_send (Tensor): Tensor to be send from current process
        * tensor_recv (Tensor): Output tensor. Its size should be ``tensor_send.numel() / get_world_size(group)``.
        * op (ReduceOp, optional): Operation used for element-wise reductions (default is ``'sum'``)
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    op = ReduceOp.get(op)
    if isinstance(tensor_send, list):
        tmp_send = torch.cat(tensor_send)
        return distributed_ext.reduce_scatter(tmp_send, tensor_recv, op, group, backend)
    else:
        return distributed_ext.reduce_scatter(tensor_send, tensor_recv, op, group, backend)


def gather(tensor_send, tensor_recv, dst=0, group=group.WORLD, backend=Backend.AUTO):
    """
    Gathers tensors from the whole group into one process.

    Arguments:
        * tensor_send (Tensor): Tensor to be send from current process
        * tensor_recv (Tensor or list[Tensor]): Output can be a tensor or a list of tensors.
          When use a tensor to recv all data from all processes' tensor_send. Its
          size should be ``get_world_size(group) * tensor_send.numel()``. When use a list of tensors,
          the size of the list should be ``get_world_size(group)`` and all the tensors in the list
          should have size ``tensor_send.numel()``. (If this process is not dst, tensor_recv will
          be ignored, should be None)
        * dst (int, optional): Destination rank (default is 0)
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """

    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    if get_rank(group) == dst:
        if isinstance(tensor_recv, list):
            world_size = get_world_size(group)
            tmp_recv = torch.empty([tensor_send.numel()*world_size],
                                   dtype=tensor_send.dtype,
                                   device=tensor_send.device)
            distributed_ext.gather(tensor_send, tmp_recv, group, backend)
            tmp_recv = tmp_recv.reshape([world_size, -1])
            for i in range(world_size):
                tensor_recv[i].copy_(tmp_recv[i])
        else:
            return distributed_ext.gather(tensor_send, tensor_recv, dst, group, backend)
    else:
        distributed_ext.gather(tensor_send, tensor_send, dst, group, backend)


def scatter(tensor_send, tensor_recv, src=0, group=group.WORLD, backend=Backend.AUTO):
    """
    Scatters tensor from one process to all processes in group.

    Arguments:
        * tensor_send (Tensor or list[Tensor]): Input can be a tensor or a list of tensors.
          When use a tensor, its size should be ``get_world_size(group) * tensor_recv.numel()``.
          When use a list of tensors, the size of the list should be ``get_world_size(group)``.
          (If this process is not src, tensor_send will be ignored, should be None)
        * tensor_recv (Tensor): Output tensor
        * src (int): Source rank (default is 0)
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
        * backend (Backend, optional): The backend to use (default is ``'auto'``)
    """
    assert _initialized, "need call distributed.init first!"
    backend = Backend.get(backend)
    if get_rank(group) == src:
        if isinstance(tensor_send, list):
            tmp_send = torch.cat(tensor_send)
            return distributed_ext.scatter(tmp_send, tensor_recv, src, group, backend)
        else:
            return distributed_ext.scatter(tensor_send, tensor_recv, src, group, backend)
    else:
        return distributed_ext.scatter(tensor_recv, tensor_recv, src, group, backend)


def barrier(group=group.WORLD):
    """
    Synchronizes all processes.

    This collective blocks processes until the whole group enters this function.

    Arguments:
        * group (ProcessGroup, optional): The process group to work on (default is ``group.WORLD``)
    """
    assert _initialized, "need call distributed.init first!"
    return distributed_ext.barrier(group)


def new_group(ranks=None):
    """
    Creates a new distributed group.

    This function requires that all processes enter this function. If parameter ``ranks``
    is not None, it should be a list of ranks which contain the current process. e.g.
    Give four processes: 0, 1, 2, 3. If create two groups which contain [0,1] and [2,3], each
    process should call:

    * process 0: ng = new_group([0, 1])
    * process 1: ng = new_group([0, 1])
    * process 2: ng = new_group([2, 3])
    * process 3: ng = new_group([2, 3])

    Arguments:
        * ranks (list[int], optional): List of ranks of new group members. If not set or None,
          will return a copy group of ``group.WORLD``

    Returns:
        An integer represents the new ProcessGroup
    """
    assert _initialized, "need call distributed.init first!"
    if ranks is None:
        ranks = list(range(get_world_size()))
    return distributed_ext.new_group(ranks)
