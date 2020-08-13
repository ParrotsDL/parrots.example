_base_ = './rpn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='/mnt/lustre/share_data/yangruichao/model_pool_data/mmdet/resnext101_64x4d-ee2c6f71.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
