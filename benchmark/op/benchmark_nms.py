import torch
import numpy
from pape.op import naive_nms
from benchmark import PAPEBenchmark


class Benchmarknms(PAPEBenchmark):

    def gen_input(self, thresh):
        np_input = numpy.load('/mnt/lustre/share/platform/benchmark/pape/nms_input.npy')
        indata_dict = {}
        indata_dict["dets"] = torch.from_numpy(np_input)
        indata_dict["dets"].requires_grad = True
        indata_dict["thresh"] = thresh
        return [], indata_dict

    def gen_op(self):
        return naive_nms


if __name__ == "__main__":
    bench = Benchmarknms()
    duration = bench.benchmark(in_args={"thresh": 0.3}, backward=False)
    print("nms time: {:.6f}".format(duration))
