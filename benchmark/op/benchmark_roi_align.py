import torch
from pape.op import RoIAlignPool
from benchmark import PAPEBenchmark
import numpy


class BenchmarkRoIAlign(PAPEBenchmark):

    def gen_input(self):
        np_input = numpy.load('/mnt/lustre/share/platform/benchmark/pape/x_test.npy')
        np_rois = numpy.load('/mnt/lustre/share/platform/benchmark/pape/rois_test.npy')
        indata_dict = {}
        indata_dict["rois"] = torch.from_numpy(np_rois)
        indata_dict["feature"] = torch.from_numpy(np_input)
        indata_dict["feature"].requires_grad = True
        indata_dict["stride"] = 1
        return [], indata_dict

    def gen_op(self, pool_h, pool_w, sampling_ratio, spatial_scale):
        return RoIAlignPool(pool_h, pool_w, sampling_ratio, spatial_scale)


if __name__ == "__main__":
    bench = BenchmarkRoIAlign()
    duration = bench.benchmark(op_args={"pool_h": 2, "pool_w": 2, "sampling_ratio": 2, "spatial_scale": 1.0},
                               backward=False)
    print("RoIAlign time: {:.6f}".format(duration))
