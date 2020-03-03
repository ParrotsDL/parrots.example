import torch
import numpy
from pape.op import soft_nms
from benchmark import PAPEBenchmark


class Benchmarksoftnms(PAPEBenchmark):

    def gen_input(self, sigma, Nt, thresh, method):
        np_input = numpy.load('/mnt/lustre/share/platform/benchmark/pape/nms_input.npy')
        indata_dict = {}
        indata_dict['dets'] = torch.from_numpy(np_input)
        indata_dict['dets'].requires_grad = True
        indata_dict['sigma'] = sigma
        indata_dict['Nt'] = Nt
        indata_dict['thresh'] = thresh
        indata_dict['method'] = method
        return [], indata_dict

    def gen_op(self):
        return soft_nms


if __name__ == "__main__":
    bench = Benchmarksoftnms()
    # test linear softnms
    duration = bench.benchmark(in_args={"sigma": 0.5, "Nt": 0.3, "thresh": 0.01, "method": 1},
                               cuda=False, backward=False)
    print("linear softnms time: {:.6f}".format(duration))

    # test gaussian softnms
    duration = bench.benchmark(in_args={"sigma": 0.5, "Nt": 0.3, "thresh": 0.01, "method": 2},
                               cuda=False, backward=False)
    print("gaussian softnms time: {:.6f}".format(duration))

    # test naive softnms
    duration = bench.benchmark(in_args={"sigma": 0.5, "Nt": 0.3, "thresh": 0.01, "method": 0},
                               cuda=False, backward=False)
    print("naive softnms time: {:.6f}".format(duration))
