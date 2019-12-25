import torch
import numpy
from pape.op import SoftmaxFocalLossFunction, SigmoidFocalLossFunction
from benchmark import PAPEBenchmark


class BenchmarkFocalLoss(PAPEBenchmark):

    def gen_input(self, losstype, weight_pos):

        np_x = numpy.load('/mnt/lustre/share/platform/benchmark/pape/pred_case.npy')
        np_y = numpy.load('/mnt/lustre/share/platform/benchmark/pape/target_case.npy')

        indata_list = [torch.from_numpy(np_x), torch.from_numpy(np_y), torch.tensor([weight_pos])]
        indata_list[0].requires_grad = True
        return indata_list, {}

    def gen_op(self, losstype, gamma, alpha, num_classes):
        loss_cls = {'sigmoid': SigmoidFocalLossFunction,
                    'softmax': SoftmaxFocalLossFunction}
        return loss_cls[losstype](gamma, alpha, num_classes)


if __name__ == "__main__":
    # test softmax
    bench = BenchmarkFocalLoss()
    duration = bench.benchmark(in_args={"losstype": 'softmax', "weight_pos": 256.},
                               op_args={"losstype": 'softmax', "gamma": 2.0, "alpha": 0.25, "num_classes": 80},
                               backward=False)
    print("softmax focal loss time: {:.6f}".format(duration))

    # test sigmoid
    duration = bench.benchmark(in_args={"losstype": 'sigmoid', "weight_pos": 256.},
                               op_args={"losstype": 'sigmoid', "gamma": 2.0, "alpha": 0.25, "num_classes": 80},
                               backward=False)
    print("sigmoid focal loss time: {:.6f}".format(duration))
