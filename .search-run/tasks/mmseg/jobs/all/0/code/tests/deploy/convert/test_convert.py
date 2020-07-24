import torch
import torch.nn as nn
from parrotsconvert.caffe import CaffeNet, BackendSet
from resnet import resnet50
import os

import numpy as np


def constant_init(parrots_model):
    for module in parrots_model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.fill_(0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.02)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(0.01)
            module.bias.fill_(0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.fill_(0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.02)


def caffe_export(net):
    net.eval()
    np.random.seed(10)
    shape = (1, 3, 224, 224)
    x = np.random.rand(*shape).astype(np.float32)
    x.tofile('in.bin')
    input = torch.from_numpy(x)

    caffe_net = CaffeNet(net, input, BackendSet.PPL, verbose=True)
    caffe_net.dump_model('./test')
    cmdstr = ('./test_model -proto-txt test.prototxt -model test.caffemodel '
              '-input in.bin --save-output')
    os.system(cmdstr)
    parrots_output = caffe_net.outputs[0].cpu().reshape(-1).numpy()
    ppl_output = np.fromfile('./test_ppl_ppl_output_0.dat', np.float32)
    assert np.allclose(parrots_output, ppl_output)
    print('Test Successfully!')

    # cmdstr = ('rm test.prototxt && rm test.caffemodel && rm test_ppl_ppl_ou'
    #            'tput_0.dat && rm in.bin')
    # os.system(cmdstr)


def test_convert():
    net = resnet50()
    constant_init(net)
    caffe_export(net)


if __name__ == '__main__':
    test_convert()
