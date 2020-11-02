_base_ = './upernet_r50_512x1024_40k_cityscapes.py'
model = dict(pretrained='/mnt/lustre/share_data/maxiaoming/mmseg_pretrain_model/resnet101_v1c_trick-e67eebb6.pth', backbone=dict(depth=101))
