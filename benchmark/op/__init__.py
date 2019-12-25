import torch
import time

from pape.utils.tensor_util import to_cuda, to_cpu
import pape.distributed as dist


class PAPEBenchmark(object):

    def __init__(self):
        pass

    def gen_input(self):
        pass

    def gen_op(self):
        pass

    def benchmark(self, in_args={}, op_args={}, warmup=10, run=100, backward=True, cuda=True):
        indata_list, indata_dict = self.gen_input(**in_args)
        op = self.gen_op(**op_args)

        if cuda:
            indata_list, indata_dict = to_cuda(indata_list), to_cuda(indata_dict)
            if hasattr(op, "cuda"):
                op = op.cuda()
            if backward:
                out = op(*indata_list, **indata_dict)
                out_grad = torch.ones_like(out).cuda()
        else:
            indata_list, indata_dict = to_cpu(indata_list), to_cpu(indata_dict)
            if hasattr(op, "cpu"):
                op = op.cpu()
            if backward:
                out = op(*indata_list, **indata_dict)
                out_grad = torch.ones_like(out)

        for i in range(warmup):
            out = op(*indata_list, **indata_dict)
            if backward:
                out.backward(out_grad)

        torch.cuda.synchronize()
        start = time.time()
        for i in range(run):
            out = op(*indata_list, **indata_dict)
            if backward:
                out.backward(out_grad)
        torch.cuda.synchronize()
        duration = (time.time() - start) / run
        return duration


class PAPEBenchmarkDistributed(PAPEBenchmark):

    def __init__(self):
        super(PAPEBenchmarkDistributed, self).__init__()
        if not dist.is_initialized():
            dist.init()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
