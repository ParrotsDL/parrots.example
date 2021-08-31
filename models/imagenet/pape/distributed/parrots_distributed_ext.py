import parrots
import parrots.comm as pcomm
import torch


mpi_chans = {}
nccl_chans = {}


class Backend(object):
    AUTO = 0
    MPI = 1
    NCCL = 2


class reduce_op(object):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    op2name = {0: "sum",
               1: "prod",
               2: "min",
               3: "max"}

    @staticmethod
    def get(op):
        return reduce_op.op2name[op]


def initialize():
    local_rank = parrots.ccomm.get_local_rank()
    torch.cuda.set_device(local_rank)
    mpi_chans[0] = pcomm.Channel.create("mpi")
    nccl_chans[0] = pcomm.Channel.create("nccl")


def finalize():
    pass


def get_rank(g):
    assert g in mpi_chans
    return pcomm.get_channel_rank(mpi_chans[g])


def get_world_size(g):
    assert g in mpi_chans
    return pcomm.get_channel_size(mpi_chans[g])


def get_local_rank():
    return parrots.ccomm.get_local_rank()


def send(tensor, dst, group, tag, backend):
    assert False, "parrots not implements this."


def recv(tensor, src, group, tag, backend):
    assert False, "parrots not implements this."


def recv_all(tensor, group, tag, backend):
    assert False, "parrots not implements this."


def broadcast(tensor, src, group, backend):
    if tensor.is_cuda and backend != Backend.MPI:
        assert group in nccl_chans
        return pcomm.bcast(tensor, src, nccl_chans[group])
    else:
        assert group in mpi_chans
        return pcomm.bcast(tensor, src, mpi_chans[group])


def all_reduce(tensor, op, group, backend):
    if isinstance(tensor, (list, tuple)):
        use_cuda = tensor[0].is_cuda
    else:
        use_cuda = tensor.is_cuda

    if use_cuda and backend != Backend.MPI:
        assert group in nccl_chans
        return pcomm.allreduce(tensor, reduce_op.get(op), nccl_chans[group])
    else:
        assert group in mpi_chans
        return pcomm.allreduce(tensor, reduce_op.get(op), mpi_chans[group])


def reduce(tensor, dst, op, group, backend):
    if tensor.is_cuda and backend != Backend.MPI:
        assert group in nccl_chans
        return pcomm.reduce(tensor, dst, reduce_op.get(op), nccl_chans[group])
    else:
        assert group in mpi_chans
        return pcomm.reduce(tensor, dst, reduce_op.get(op), mpi_chans[group])


def all_gather(tensor_send, tensor_recv, group, backend):
    if tensor_send.is_cuda and backend != Backend.MPI:
        assert group in nccl_chans
        return pcomm.allgather(tensor_send, tensor_recv, nccl_chans[group])
    else:
        assert group in mpi_chans
        return pcomm.allgather(tensor_send, tensor_recv, mpi_chans[group])


def reduce_scatter(tensor_send, tensor_recv, op, group, backend):
    if tensor_recv.is_cuda and backend != Backend.MPI:
        assert group in nccl_chans
        return pcomm.reducescatter(tensor_send, tensor_recv, reduce_op.get(op), nccl_chans[group])
    else:
        assert group in mpi_chans
        return pcomm.reducescatter(tensor_send, tensor_recv, reduce_op.get(op), mpi_chans[group])


def gather(tensor_send, tensor_recv, dst, group, backend):
    if tensor_send.is_cuda:
        assert backend != Backend.MPI
        assert group in nccl_chans
        return pcomm.gather(tensor_send, tensor_recv, dst, nccl_chans[group])
    else:
        assert backend != Backend.NCCL
        assert group in mpi_chans
        return pcomm.gather(tensor_send, tensor_recv, dst, mpi_chans[group])


def scatter(tensor_send, tensor_recv, src, group, backend):
    if tensor_recv.is_cuda:
        assert backend != Backend.MPI
        assert group in nccl_chans
        return pcomm.scatter(tensor_send, tensor_recv, src, nccl_chans[group])
    else:
        assert backend != Backend.NCCL
        assert group in mpi_chans
        return pcomm.scatter(tensor_send, tensor_recv, src, mpi_chans[group])


def barrier(group):
    assert group in mpi_chans
    assert group in nccl_chans
    pcomm.barrier(mpi_chans[group])
    pcomm.barrier(nccl_chans[group])


def synchronize(group):
    assert group in mpi_chans
    assert group in nccl_chans
    pcomm.sync(mpi_chans[group])
    pcomm.sync(nccl_chans[group])


def new_group(ranks):
    new_group_id = len(mpi_chans)
    assert new_group_id not in mpi_chans
    pcomm.set_default_backend("mpi")
    mpi_chans[new_group_id] = pcomm.new_channel(ranks)
    pcomm.set_default_backend("nccl")
    nccl_chans[new_group_id] = pcomm.new_channel(ranks)
    return new_group_id
