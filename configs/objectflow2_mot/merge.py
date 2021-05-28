epoch_size = 10000
conv_module = dict(type='ConvModule', kernel_size=3, padding=1, bias=False, norm_cfg=dict(type='MMSyncBN'))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10*epoch_size, 20*epoch_size])
evaluation = dict(interval=epoch_size,
                  metric=dict(bbox_head=dict(metric='RatP', target_prec=0.95)))

# evaluation = dict(
#     interval=31*epoch_size,
#     metric=dict(
#         bbox_head=dict(metric='RatP', target_prec=0.99),
#         cls_head=dict(metric='ClsTest', target_prec=0.95, GT_MODE=False),
#     ))

max_iters = epoch_size * 30
workflow = [('train', max_iters)]
# task_groups = {'debug': [0]}
# task_prefix = {}
task_groups = {'face': [i for i in range(0, 8)],
               'body': [i for i in range(8, 16)],
               'facebody': [i for i in range(0, 16)],
               'hoi': [i for i in range(16, 24)],
               'reid': [i for i in range(24, 32)],
               'mot15': [i for i in range(24, 25)],
               'mot16': [i for i in range(25, 26)],
               'mot20': [i for i in range(26, 27)],
               'prw': [i for i in range(27, 28)],
               'ipc_v2': [i for i in range(28, 32)],
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
load_from = None
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
        num_classes=2,
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
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    cls_head=dict(
        type='FMHead',
        gt_jitter=0.2,
        in_channels=32,
        feat_channels=32,
        feat_out_channels=32,
        shared_convs_cfg=dict(
            type='ChooseOneFromListInTensorOut',
            choose_ind=0,
            conv_list=[conv_module, conv_module, conv_module]),
        branch_cfg_list=[
            dict(
                name='track_id',
                base_task='full_body',
                type='aiocls',
                net=dict(
                    type='TensorInListOutRefine',
                    conv_list=[
                        dict(type='ConvModule', kernel_size=1),
                        dict(type='ConvModule', kernel_size=1, act_cfg=None)
                    ]),
                cls_channels=32,
                loss=dict(type='CrossEntropyLoss', loss_weight=0.5)),
        ]),
)
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
    det_score_thrd=[0.5, 0.5],
    cls_score_thrd=[0.5, 0.5],
    rel_cls_score_thrd=0.5,
    hoi_rel_thrd=1e-2,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=100,
    valid_range = [460, 150, 1590, 870],# [460, 150, 1590, 870]
    cls_head=dict(
        neck=dict(
            type='DCONV_FPN',
            in_channels=[24, 32, 96, 160],
            out_channels=32,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5,
            norm_cfg=dict(type='MMSyncBN')),
    ),
    reid_cfg = dict(
        conf_thres=0.8,#opt.conf_thres
        crossline_most_lost=60,
        track_buffer=60,
        return_gt_box=True,
        test_img_save_dir="./debug/image/",
        result_filename = "./debug/output_2.txt"
    )
)

# dataset set settings
dataset_type = 'CustomTaskDataset'
gaosi_root = '/mnt/lustre/fangkairen/SomePklData/child_train_test/gaosi_room/'
root_du = '/mnt/lustrenew/dutianyuan/detection_data/'

detect_task = dict(face=0, full_body=1)
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
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file='/mnt/lustre/share/zhangqiming/data/mot/HKJCV1testfilter_20201026.pkl',
    #     img_prefix='/mnt/lustre/share/zhangqiming/data/mot/pack_HKJCV1/frames/',
    #     with_ignore_bboxes=True,
    #     detect_task=detect_task,
    #     extra_task=extra_task,
    #     pipeline=test_pipeline,
    #     test_cfg=test_cfg),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file='/mnt/lustre/share/zhangqiming/data/mot/HKJCV2testfilter_20201029.pkl',
    #     img_prefix='/mnt/lustre/share/zhangqiming/data/horce_race_pan/',
    #     with_ignore_bboxes=True,
    #     detect_task=detect_task,
    #     extra_task=extra_task,
    #     pipeline=test_pipeline,
    #     test_cfg=test_cfg),
    dict(
       type=dataset_type,
        ann_file=mot_root + 'mot_test.pkl',
        img_prefix='sh416:s3://mot/mnt/lustre/share/zhangqiming/HKJC_test/frames/',
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
            mot_root + 'mot/HKJCV1_20201026.pkl',
            mot_root + 'mot/HKJCV2_20201029.pkl',
        ],
        img_prefix=[
            'sh1984:s3://parrots_model_data/objectflow_mot/data/mot/pack_HKJCV1/frames/',
            'sh1984:s3://parrots_model_data/objectflow_mot/data/horce_race_pan/',
	    ],
        with_ignore_bboxes=True,  # gt_bboxes_ignore
        detect_task=detect_task,
        extra_task=extra_task,
        pipeline=train_pipeline),
    val=test_datasets,
    test=test_datasets)
