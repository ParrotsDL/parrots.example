_base_ = [
    './_base_/schedules/schedule_sgd_600e.py', './_base_/default_runtime.py'
]
model = dict(
    type='PSENet',
    pretrained='/mnt/lustre/share_data/parrots_model_data/mmocr/premodel/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPNF',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_type='concat'),
    bbox_head=dict(
        type='PSEHead',
        text_repr_type='poly',
        in_channels=[256],
        out_channels=7,
        loss=dict(type='PSELoss')))
train_cfg = None
test_cfg = None

dataset_type = 'IcdarDataset'
data_root = '/mnt/lustre/share_data/parrots_model_data/mmocr/data/ctw1500/'
ceph_data_root = 's3://parrots_model_data/mmocr/data/ctw1500/'
# img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='ScaleAspectJitter',
        img_scale=[(3000, 736)],  # unused
        ratio_range=(0.5, 3),
        aspect_ratio_range=(1, 1),
        multiscale_mode='value',
        pre_long_size_bound=1280,
        final_short_size_bound=640,
        resize_type='up_lower_bound',
        keep_ratio=False),
    dict(
        type='GenPANetTargets',
        shrink_ratio=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotate'),
    dict(type='RandomCropInstances', target_size=(640, 640)),
    dict(type='Pad', size_divisor=32),
    dict(type='PANetFormatBundle', visualize=False),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_effective_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_training.json',
        # for debugging top k imgs
        # select_first_k=2,
        img_prefix=data_root + '/imgs',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        # select_first_k=2,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        # select_first_k=2,
        pipeline=test_pipeline))

evaluation = dict(interval=100, metric='hmean')
