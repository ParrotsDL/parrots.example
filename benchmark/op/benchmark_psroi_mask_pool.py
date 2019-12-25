import torch
import numpy
from pape.op import PSRoIMaskPool
from benchmark import PAPEBenchmark


class BenchmarkPSRoIMaskPool(PAPEBenchmark):

    def gen_input(self):
        np_input = numpy.load('/mnt/lustre/share/platform/benchmark/pape/ps_x_test.npy')
        np_rois = numpy.load('/mnt/lustre/share/platform/benchmark/pape/ps_rois_test.npy')
        indata_dict = {}
        indata_dict["rois"] = torch.from_numpy(np_rois)
        indata_dict["features"] = torch.from_numpy(np_input)
        indata_dict["features"].requires_grad = True
        indata_dict["stride"] = 1
        return [], indata_dict

    def gen_op(self, group_size, roi_scale, bin_scale, output_dim=None, spatial_scale=None):
        return PSRoIMaskPool(group_size, roi_scale, bin_scale)


if __name__ == "__main__":
    bench = BenchmarkPSRoIMaskPool()
    duration = bench.benchmark(op_args={"group_size": 4, "spatial_scale": 1.0, "roi_scale": 1.5,
                                        "bin_scale": 2, "output_dim": 16}, backward=False)
    print("PSRoIMaskPool time: {:.6f}".format(duration))
