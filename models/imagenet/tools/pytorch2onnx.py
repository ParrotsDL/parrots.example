'''
    Before use: export PYTHONPATH=path_to_onnx_simplifier;parrots.example/models/imagenet/models;
    run: python modelConvert.py --shape 1 3 224 224 --model_path camb_pth/resnet101/resnet101_best_float.pth --simplify

    Args:
        shape: the input shape of the model, default is [224, 224]
        model_path: the path of a pytorch float model, format should be like "camb_pth/resnet101/resnet101_best_float.pth", 
        script will parse to use resnet101 to instance a model
        simplify: use onnx-simplify to simplify the model or not
'''


import torch
import argparse
import onnx
from onnxsim import simplify
from resnet import resnet101, resnet50
from mobile_v2 import mobile_v2
from inception_v3 import inception_v3
from alexnet import alexnet
from googlenet import googlenet
from vgg import vgg16

model_dict = {'alexnet': alexnet, 'googlenet':googlenet, \
    'inception_v3': inception_v3, 'mobile_v2':mobile_v2, \
    'resnet101':resnet101, 'resnet50':resnet50, 'vgg':vgg16}
def parse_args():
    parser = argparse.ArgumentParser(description='convert model to onnx')
    parser.add_argument('--shape', type=int, nargs='+',
        default=[224, 224], help='shape of the model input')
    # The model_path should be xxx/xxx/${model_name}/xxx.pth,
    # for the ${model_name} will be used to verify which model to initialize,
    # eg: camb_pth/resnet50/resnet50_best.pth, then "resnet50" will be used.
    parser.add_argument('--model_path', type=str,
        help='the origin path of the model')
    parser.add_argument('--simplify', dest='simplify', action='store_true',
        help='flag if the onnx model need simplify')
    args = parser.parse_args()
    return args

def parrots2Onnx(input, model_path, onnx_path, do_simplify=False):
    print(model_path.split('/')[-2])
    model = model_dict[model_path.split('/')[-2]]().cuda()
    model_ins = torch.load(model_path)
    if 'state_dict' in model_ins.keys():
        model_ins = model_ins['state_dict']
    new_model_ins = dict()
    for key in model_ins:
        if key[0:7] == "module.":
            new_key = key[7:]
            new_model_ins[new_key] = model_ins[key]
        else:
            new_key = key
            new_model_ins[new_key] = model_ins[key]
    model.load_state_dict(new_model_ins)
    model = model.cuda()
    print(onnx_path)
    torch.onnx.export(model, input, onnx_path, verbose=True, opset_version=11)

    if do_simplify:
        onnx_origin = onnx.load(onnx_path)
        onnx_simplify, check = simplify(onnx_origin)
        if check:
            onnx_simplify_path = onnx_path.split('.')[0] + '_simplify.onnx'
            onnx.save(onnx_simplify, onnx_simplify_path)
            print(f'Successfully simplified ONNX model: {onnx_simplify_path}')
        else:
            print(f'Failed to simplify ONNX model.')


def main():
    args = parse_args()
    shape = args.shape
    print(shape)
    simplify_flag = args.simplify
    dummy_input = torch.randn(shape).cuda()
    model_path = args.model_path
    onnx_path = model_path.split('.')[0] + '.onnx'
    parrots2Onnx(dummy_input, model_path, onnx_path, simplify_flag)

if __name__ == '__main__':
    main()
