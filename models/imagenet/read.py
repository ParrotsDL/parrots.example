import torch
import models
import copy
import argparse

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--pth_path', default='',
                    type=str, help='path to checkpoint file')
args = parser.parse_args()

args.pth_path = "/share1/fengsibo/parrots.example/models/imagenet/checkpoints/resnet50_1019/resnet50_ckpt_epoch_16.pth"



def load_checkpoint(model, map_location=''):
    state_dict = torch.load(args.pth_path, map_location='cpu')
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict['state_dict'])
    for p in state_dict:
        print(p, state_dict[p])


def main():
    model = models.resnet50()
    # model = model.cuda()
    # model = model.to_memory_format(torch.channels_last)

    # input = torch.randn(2, 3, 224, 224, requires_grad=True).cuda()
    # input = input.contiguous(torch.channels_last)
    # out = model(input)

    load_checkpoint(model)


if __name__ == "__main__":
    main()