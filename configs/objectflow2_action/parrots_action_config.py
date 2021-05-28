base_root = '../../models/objectflow2_action/mmdetection/configs/_base_/'
_base_ = [
    base_root + '/default_runtime.py',
]
# optimizer
optimizer = dict(type='Adam', lr=4.5e-4)
find_unused_parameters=True

# load_from='/mnt/lustre/share/chenguangqi/hoi/transfer_1984/final_student_exp07.pth' # for SH38
load_from='/mnt/lustre/share_data/parrots_model_ckpt/objectflow_action/final_student_exp07.pth' # for 1984

# model settings
# normalize = dict(type='MMSyncBN')
normalize = dict(type='SyncBN')
model = dict(
    type='PPDM',
    backbone=dict(
        type='MobileNetV2_prun',
        #input_channel=8,
        last_feat_channel=160,
        img_channel=1,
        norm_eval=False,
        out_indices=(0, 1, 2, 3),
        normalize=normalize),
    neck=dict(
        type='DCONV_FPN',
        in_channels=[8, 16, 40, 64],
        out_channels=16,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=normalize,
        relu_before_extra_convs=True,
        act_cfg=dict(type='ReLU')),
    hoi_head=dict(
        type='HOIHead',
        in_channels=16,
        feat_channels=16,
        num_classes_object=2,
        num_classes_verb=3,
        feature_stride=8,
        feature_level=0,  # second level in FPN
        max_objs=128,
        max_rels=64,
        reg_center_offset=False,
        with_bbox_head=False,
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=1.0,
            reduction='mean',
            loss_weight=1.0),
        loss_reg=dict(type='L1Loss',
            reduction='mean',
            loss_weight=1.0)),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=3,
        in_channels=16,
        stacked_convs=2, # 2
        feat_channels=16, # 16
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[3],
            ratios=[0.5, 1.0, 2],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=3.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=3.0))
)
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    det_only=False,
    nms_pre=40,
    min_bbox_size=0,
    score_thr=0.13,
    nms=dict(type='nms', iou_thr=0.3),
    max_per_img=20,
    max_rel=1)
# dataset settings
dataset_type = 'CustomTaskDataset'
data_root = '/mnt/lustre/share_data/parrots_model_data/objectflow_action/hoi_new_formart/'
# data_cgq = '/mnt/lustre/share/chenguangqi/hoi/'

# task select
detect_task = {"face": 0, "phone": 1, "cup": 2} # start from 0
extra_task = ["track_id", 'action'] # list:"track_id","action"...

img_norm_cfg = dict(mean=[0.], std=[1.], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
    dict(type='Resize', img_scale=(448, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='DefaultFormatBundleTask'),
    dict(
        type='CollectTask',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'],
        output_extra_anns=True,
        extra_keys=extra_task)
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root + 'trainBadCase_20200716_action.pkl',
            data_root + 'clean_cds_40w.pkl',
            data_root + 'dms_20200716_negsmok.pkl',
            data_root + 'carIndoorOutdoor_20200716_action.pkl',
            data_root + 'objectproNormalTrain_20200918.pkl',
        ],
        img_prefix=[
            'sh1984:s3://parrots_model_data/objectflow_action/detection_dataset_zqm/hoi_dataset/',
            'sh1984:s3://parrots_model_data/objectflow_action/detection_dataset_zqm/hoi_dataset/',
            'sh1984:s3://parrots_model_data/objectflow_action/detection_dataset_zqm/hoi_dataset/',
            'sh1984:s3://parrots_model_data/objectflow_action/detection_dataset_zqm/hoi_dataset/',
            'sh1984:s3://parrots_model_data/objectflow_action/detection_dataset_zqm/hoi_dataset/rotated_pro_image/',
        ],
        with_ignore_bboxes=True,  # gt_bboxes_ignore
        detect_task=detect_task,
        extra_task=extra_task,
        pipeline=train_pipeline),
    val=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'test_15k.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_action/detection_dataset_zqm/hoi_dataset/images/',
            with_ignore_bboxes=True,
            detect_task=detect_task,
            extra_task=extra_task,
            pipeline=test_pipeline),
        ],
    test=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'test_15k.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_action/detection_dataset_zqm/hoi_dataset/images/',
            with_ignore_bboxes=True,
            detect_task=detect_task,
            extra_task=extra_task,
            pipeline=test_pipeline),
    ]
    )

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20, 25])
total_epochs = 30

log_config = dict(interval=10,
                  hooks=[dict(type='TensorboardLoggerHook'), 
                         dict(type='TextLoggerHook')]
                 )

evaluation = dict(
    interval=1,
    metric=dict(
        hoi_head=dict(metric='HOIRatP', target_prec=0.99, hoi_eval=False),
    )
)

