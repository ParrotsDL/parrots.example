_base_ = './mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis.py'
model = dict(pretrained='/mnt/lustre/share_data/yangruichao/model_pool_data/mmdet/resnet101-5d3b4d8f.pth', backbone=dict(depth=101))
