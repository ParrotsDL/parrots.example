import torch
from pape.op import gpu_iou_overlap
from benchmark import PAPEBenchmark
import numpy


class BenchmarkIOUOverlap(PAPEBenchmark):

    def gen_input(self, b1, b2):
        indata_dict = {}
        # numpy.save('iou_b1', b1)
        # numpy.save('iou_b2', b2)
        indata_dict["b1"] = torch.FloatTensor(b1)
        indata_dict["b2"] = torch.FloatTensor(b2)
        return [], indata_dict

    def gen_op(self):
        return gpu_iou_overlap


if __name__ == "__main__":
    bench = BenchmarkIOUOverlap()
    duration = bench.benchmark(in_args={
        "b1": numpy.load('/mnt/lustre/share/platform/benchmark/pape/iou_b1.npy'),
        "b2": numpy.load('/mnt/lustre/share/platform/benchmark/pape/iou_b2.npy')}, backward=False)
    print("iou overlap time: {:.6f}".format(duration))
