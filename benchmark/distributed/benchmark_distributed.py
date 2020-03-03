import torch
import pape.distributed as dist
from benchmark import PAPEBenchmarkDistributed


class BenchmarkDistributedBroadcast(PAPEBenchmarkDistributed):

    def gen_input(self, tshape, root=0, group=dist.group.WORLD, backend="auto"):
        rank = dist.get_rank()
        x_all = torch.rand(tshape)
        if root == rank:
            x = x_all
        else:
            x = torch.ones(tshape)
        in_data_list = [x, root, group, backend]
        return in_data_list, {}

    def gen_op(self):
        return dist.broadcast


class BenchmarkDistributedAllReduce(PAPEBenchmarkDistributed):

    def gen_input(self, tshape, op="sum", group=dist.group.WORLD, backend="auto"):
        rank = dist.get_rank()

        x_all = torch.rand(tshape)
        xshape = [4, tshape[0]//4]
        for i in range(1, len(tshape)):
            xshape.append(tshape[i])

        x = x_all.clone().reshape(xshape)[rank]
        in_data_list = [x, op, group, backend]
        return in_data_list, {}

    def gen_op(self):
        return dist.all_reduce


class BenchmarkDistributedReduce(PAPEBenchmarkDistributed):

    def gen_input(self, tshape, root=0, op="sum", group=dist.group.WORLD, backend="auto"):
        rank = dist.get_rank()

        x_all = torch.rand(tshape)
        xshape = [4, tshape[0]//4]
        for i in range(1, len(tshape)):
            xshape.append(tshape[i])

        x = x_all.clone().reshape(xshape)[rank]
        in_data_list = [x, root, op, group, backend]
        return in_data_list, {}

    def gen_op(self):
        return dist.reduce


class BenchmarkDistributedAllGather(PAPEBenchmarkDistributed):

    def gen_input(self, tshape, group=dist.group.WORLD, backend="auto"):
        rank = dist.get_rank()

        x_all = torch.rand(tshape)
        xshape = [4, tshape[0]//4]
        for i in range(1, len(tshape)):
            xshape.append(tshape[i])

        x = x_all.clone().reshape(xshape)[rank]
        y = torch.ones_like(x_all)
        in_data_list = [x, y, group, backend]
        return in_data_list, {}

    def gen_op(self):
        return dist.all_gather


class BenchmarkDistributedReduceScatter(PAPEBenchmarkDistributed):

    def gen_input(self, tshape, op="sum", group=dist.group.WORLD, backend="auto"):
        rank = dist.get_rank()

        x_all = torch.rand(tshape)
        xshape = [4, tshape[0]//4]
        for i in range(1, len(tshape)):
            xshape.append(tshape[i])

        x = x_all.clone().reshape(xshape)[rank]
        y = torch.ones_like(x[0:1])
        in_data_list = [x, y, op, group, backend]
        return in_data_list, {}

    def gen_op(self):
        return dist.reduce_scatter


class BenchmarkDistributedGather(PAPEBenchmarkDistributed):

    def gen_input(self, tshape, root=0, group=dist.group.WORLD, backend="auto"):
        rank = dist.get_rank()

        x_all = torch.rand(tshape)
        xshape = [4, tshape[0]//4]
        for i in range(1, len(tshape)):
            xshape.append(tshape[i])

        x = x_all.clone().reshape(xshape)[rank]
        y = torch.ones_like(x_all)
        in_data_list = [x, y, root, group, backend]
        return in_data_list, {}

    def gen_op(self):
        return dist.gather


class BenchmarkDistributedScatter(PAPEBenchmarkDistributed):

    def gen_input(self, tshape, root=0, group=dist.group.WORLD, backend="auto"):
        rank = dist.get_rank()

        x_all = torch.rand(tshape)
        xshape = [4, tshape[0]//4]
        for i in range(1, len(tshape)):
            xshape.append(tshape[i])

        x = x_all.clone().reshape(xshape)[rank]
        y = torch.ones_like(x)
        in_data_list = [x_all, y, root, group, backend]
        return in_data_list, {}

    def gen_op(self):
        return dist.scatter


# To run this test: srun -p Platform -n4 --gres=gpu:8 --ntasks-per-node=8 python -m benchmark.benchmark_distributed
if __name__ == "__main__":
    bench = BenchmarkDistributedBroadcast()
    duration = bench.benchmark(in_args={"tshape": (16, 3, 2, 3)}, backward=False)
    print("broadcast time: {:.6f}".format(duration))

    bench = BenchmarkDistributedAllReduce()
    duration = bench.benchmark(in_args={"tshape": (16, 3, 2, 3)}, backward=False)
    print("all reduce time: {:.6f}".format(duration))

    bench = BenchmarkDistributedReduce()
    duration = bench.benchmark(in_args={"tshape": (16, 3, 2, 3)}, backward=False)
    print("reduce time: {:.6f}".format(duration))

    bench = BenchmarkDistributedAllGather()
    duration = bench.benchmark(in_args={"tshape": (16, 3, 2, 3)}, backward=False)
    print("all gather time: {:.6f}".format(duration))

    bench = BenchmarkDistributedReduceScatter()
    duration = bench.benchmark(in_args={"tshape": (16, 3, 2, 3)}, backward=False)
    print("reduce scatter time: {:.6f}".format(duration))

    bench = BenchmarkDistributedGather()
    duration = bench.benchmark(in_args={"tshape": (16, 3, 2, 3)}, backward=False)
    print("gather time: {:.6f}".format(duration))

    bench = BenchmarkDistributedScatter()
    duration = bench.benchmark(in_args={"tshape": (16, 3, 2, 3)}, backward=False)
    print("scatter time: {:.6f}".format(duration))
