_base_ = './grid_rcnn_r50_fpn_gn-head_2x_coco.py'

model = dict(pretrained='/mnt/lustre/share_data/yangruichao/model_pool_data/mmdet/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))
