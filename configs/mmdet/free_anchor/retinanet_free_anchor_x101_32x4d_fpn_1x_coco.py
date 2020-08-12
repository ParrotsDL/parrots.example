_base_ = './retinanet_free_anchor_r50_fpn_1x_coco.py'
model = dict(
    pretrained='/mnt/lustre/share_data/yangruichao/model_pool_data/mmdet/resnext101_32x4d-a5af3160.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'))
