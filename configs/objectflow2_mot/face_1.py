epoch_size = 10000
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7*epoch_size, 14*epoch_size])
evaluation = dict(interval=epoch_size,
                  metric=dict(bbox_head=dict(metric='RatP', target_prec=0.99)))
max_iters = epoch_size * 30
workflow = [('train', max_iters)]
# task_groups = {'debug': [0]}
# task_prefix = {}

task_groups = {'face': [i for i in range(0, 8)],
               'body': [i for i in range(8, 16)],
               'facebody': [i for i in range(0, 16)],
               'hoi': [i for i in range(16, 24)],
               'reid': [i for i in range(24, 40)],
               'mot15': [i for i in range(24, 25)],
               'mot16': [i for i in range(25, 26)],
               'mot20': [i for i in range(26, 27)],
               'prw': [i for i in range(27, 28)],
               'ipc_v2': [i for i in range(28, 32)],
               'hkjc_0': [i for i in range(32, 33)],
               'hkjc_1': [i for i in range(33, 34)],
               'hkjc_2': [i for i in range(34, 35)],
               'hkjc_3': [i for i in range(35, 36)],
               'hkjc_4': [i for i in range(36, 37)],
               'hkjc_5': [i for i in range(37, 38)],
               'hkjc_6': [i for i in range(38, 39)],
               'hkjc_7': [i for i in range(39, 40)],
               }
task_prefix = {'neck': 'facebody',
               'bbox_head.cls_convs': 'facebody',
               'bbox_head.reg_convs': 'facebody',
               'bbox_head.retina_reg': 'facebody',
               'bbox_head.retina_cls': 'face'}

checkpoint_config = dict(interval=epoch_size)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/mnt/lustre/share_data/parrots_model_ckpt/objectflow_mot/facebodyhoireid_20_3/_task0/iter_140000.pth"
resume_from = None
model = dict(
    type='RetinaNetMultiTask',
    backbone=dict(
        type='MobileNetV2_ImgNet',
        last_feat_channel=160,
        img_channel=3,
        out_indices=(0, 1, 2, 3),
        normalize=dict(type='MMSyncBN')),
    neck=dict(
        type='DCONV_FPN',
        in_channels=[24, 32, 96, 160],
        out_channels=32,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='MMSyncBN')),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=32,
        stacked_convs=4,
        feat_channels=32,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
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
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=100,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=100)
# dataset set settings
dataset_type = 'CustomTaskDataset'
gaosi_root = '/mnt/lustre/share/shenzan/motpkl/child_train_test/gaosi_room/'
root_du = 'sh1985:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/'
detect_task = dict(face=0)
extra_task = []

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(type='Resize', img_scale=(960, 544), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectTask',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'],
        extra_keys=[])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color', file_client_args=dict(backend='petrel')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 544),
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

mot_root = '/mnt/lustre/share_data/parrots_model_data/objectflow_mot/motpkl/'
test_datasets = [
    dict(
        type='CustomTaskDataset',
        ann_file=mot_root + 'mot/HKJCV1testfilter_20201026.pkl',
        img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/data/mot/pack_HKJCV1/frames/',
        with_ignore_bboxes=True,
        detect_task=detect_task,
        extra_task=extra_task,
        pipeline=test_pipeline,
        test_cfg=test_cfg),
    dict(
        type='CustomTaskDataset',
        ann_file=mot_root + 'mot/HKJCV2testfilter_20201029.pkl',
        img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/data/horce_race_pan/',
        with_ignore_bboxes=True,
        detect_task=detect_task,
        extra_task=extra_task,
        pipeline=test_pipeline,
        test_cfg=test_cfg),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=[
            mot_root + 'face_detection/new_annos/task_14484.pkl',
            # mot_root + 'face_detection/new_annos/task_14411.pkl', 
            mot_root + 'face_detection/new_annos/shangchang2768_20200605.pkl',
            mot_root + 'face_detection/new_annos/shangchang2993_20200605.pkl',
            mot_root + 'face_detection/new_annos/shangchang3572_20200605.pkl',
            # mot_root + 'face_detection/new_annos/ipc.pkl',
            mot_root + 'face_detection/new_annos/crowdhuman_train.pkl',
            # mot_root + 'face_detection/new_annos/crowdhuman_val.pkl',
            # mot_root + 'face_detection/new_annos/task_2211.pkl',
            # gaosi_root + 'child_reanno_12447.pkl',
            # gaosi_root + 'child_reanno_12455.pkl',
            # gaosi_root + 'child_reanno_12456.pkl',
            # gaosi_root + 'child_reanno_12458.pkl',
            # gaosi_root + 'child_reanno_12459.pkl',
            # gaosi_root + 'child_reanno_12460.pkl',
            # gaosi_root + 'child_reanno_12774.pkl',
            # gaosi_root + 'child_reanno_12911.pkl',
            mot_root + 'mot/HKJCV1_20201026.pkl',
            mot_root + 'mot/HKJCV2_20201029.pkl',
        ],
        img_prefix=[
            'sh1984:s3://parrots_model_data/objectflow_mot/wangliming/data/detection_data/shangfei_mendian/task_14484/Images/',
            # '/mnt/lustre/share/zhangqiming/wangliming/data//detection_data/task_14411/Images/',
            'sh1984:s3://parrots_model_data/objectflow_mot/wangliming/data/data_from_applyment_mask_data/zhengshuai_0609/task_2768/Images/',
            'sh1984:s3://parrots_model_data/objectflow_mot/wangliming/data/data_from_applyment_mask_data/zhengshuai_0609/task_2993/Images/',
            'sh1984:s3://parrots_model_data/objectflow_mot/wangliming/data/data_from_applyment_mask_data/zhengshuai_0609/task_3572/Images/',
            # '/mnt/lustre/share/zhangqiming/wangliming/backup_data/ipc_data/',
            's3://parrots_model_data/objectflow_mot/wangliming/data/detection_data/CrowdHuman/CrowdHuman_train/Images/',
            # '/mnt/lustre/share/zhangqiming/wangliming/data//detection_data/CrowdHuman//CrowdHuman_val/Images/',
            # '/mnt/lustre/share/zhangqiming/wangliming/data/data_from_applyment_mask_data/zhengshuai_0610_face/task_2211/Images',
            # root_du + 'body_data/reanno_dataset/reanno_12447',
            # root_du + 'body_data/reanno_dataset/reanno_12455',
            # root_du + 'body_data/reanno_dataset/reanno_12456',
            # root_du + 'body_data/reanno_dataset/reanno_12458',
            # root_du + 'body_data/reanno_dataset/reanno_12459',
            # root_du + 'body_data/reanno_dataset/reanno_12460',
            # root_du + 'body_data/reanno_dataset/reanno_12774',
            # root_du + 'body_data/reanno_dataset/reanno_12911',
            'sh1984:s3://parrots_model_data/objectflow_mot/data/mot/pack_HKJCV1/frames/',
            'sh1984:s3://parrots_model_data/objectflow_mot/data/horce_race_pan/',
	    ],
        with_ignore_bboxes=True,  # gt_bboxes_ignore
        detect_task=detect_task,
        extra_task=extra_task,
        pipeline=train_pipeline),
    val=test_datasets,
    test=test_datasets)
