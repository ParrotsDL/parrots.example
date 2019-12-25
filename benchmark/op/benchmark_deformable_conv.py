import torch
from pape.op import DeformableConv, DeformConv2d, DeformableConvInOne
from benchmark import PAPEBenchmark
import numpy


class BenchmarkDeformconv(PAPEBenchmark):

    def gen_input(self, dtype):
        np_x = numpy.load('/mnt/lustre/share/platform/benchmark/pape/x_deform_test.npy')
        np_offset = numpy.load('/mnt/lustre/share/platform/benchmark/pape/offset_deform_test.npy')
        indata_dict = {}
        indata_dict['input'] = torch.from_numpy(np_x)
        indata_dict['input'].requires_grad = True
        if dtype == 'deformconv':
            indata_dict['offset'] = torch.from_numpy(np_offset)
        return [], indata_dict

    def gen_op(self, dtype, in_channels, out_channels, kernel_size,
               stride=1, padding=0, dilation=2, groups=1, bias=True, deform_groups=1):
        if dtype == 'deformconv':
            return DeformableConv(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, groups, deform_groups)
        elif dtype == 'deformconv2d':
            return DeformConv2d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias, deform_groups)
        elif dtype == 'deformconvinone':
            return DeformableConvInOne(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias, deform_groups)


if __name__ == "__main__":
    bench = BenchmarkDeformconv()

    # test DeformableConv
    duration = bench.benchmark(in_args={"dtype": 'deformconv'}, op_args={"dtype": 'deformconv', "in_channels": 512,
                               "out_channels": 512, "kernel_size": 3, "padding": 1})
    print("DeformableConv time: {:.6f}".format(duration))

    # test DeformConv2d
    duration = bench.benchmark(in_args={"dtype": 'deformconv2d'}, op_args={"dtype": 'deformconv2d', "in_channels": 512,
                                                                           "out_channels": 1024, "kernel_size": 3})
    print("DeformConv2d time: {:.6f}".format(duration))

    # test DeformableConvInOne
    duration = bench.benchmark(in_args={"dtype": 'deformconvinone'},
                               op_args={"dtype": 'deformconvinone', "in_channels": 512, "out_channels": 1024,
                                        "kernel_size": 3, "bias": False})
    print("DeformableConvInOne time: {:.6f}".format(duration))
