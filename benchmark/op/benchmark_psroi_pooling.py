import torch
import numpy
from pape.op import PSRoIPool
from benchmark import PAPEBenchmark


class BenchmarkPSRoIPool(PAPEBenchmark):

    def gen_input(self):
        np_input = numpy.load('/mnt/lustre/share/platform/benchmark/pape/ps_x_test.npy')
        np_rois = numpy.load('/mnt/lustre/share/platform/benchmark/pape/ps_rois_test.npy')
        indata_dict = {}
        indata_dict["rois"] = torch.from_numpy(np_rois)
        indata_dict["features"] = torch.from_numpy(np_input)
        indata_dict["features"].requires_grad = True
        indata_dict["stride"] = 1
        return [], indata_dict

    def gen_op(self, group_size, output_dim=None, spatial_scale=None):
        return PSRoIPool(group_size)


if __name__ == "__main__":
    bench = BenchmarkPSRoIPool()
    duration = bench.benchmark(op_args={"group_size": 4, "spatial_scale": 1.0, "output_dim": 16}, backward=False)
    print("PSRoIPool time: {:.6f}".format(duration))
