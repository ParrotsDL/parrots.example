_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(pretrained='/mnt/lustre/share_data/yangruichao/model_pool_data/mmdet/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
