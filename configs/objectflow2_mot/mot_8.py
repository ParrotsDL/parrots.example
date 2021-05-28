epoch_size = 10000
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7*epoch_size, 14*epoch_size])
evaluation = dict(
    interval=31*epoch_size,
    metric=dict(
        cls_head=dict(metric='ClsTest', target_prec=0.95, GT_MODE=True),
    ))

max_iters = epoch_size * 30
workflow = [('train', max_iters)]
fix_prefix = []

# task_groups = {'reid': [i for i in range(0, 1)]}
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
task_prefix = {'neck.lateral_convs.0': 'reid',
               'neck.lateral_convs.1': 'reid',
               'neck.lateral_convs.2': 'reid',
               'neck.d_conv': 'reid',
               'neck.fpn_convs.0': 'reid',
               'neck.fpn_convs.1': None,
               'neck.fpn_convs.2': None,
               'neck.fpn_convs.3': None,
               'neck.fpn_convs.4': None,
               'cls_head.shared_convs': 'reid',
               'cls_head.track_id': 'reid',
               'cls_head.fc': 'hkjc_2'}

checkpoint_config = dict(interval=epoch_size)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/mnt/lustre/share_data/parrots_model_ckpt/objectflow_mot/facebodyhoireid_20_3/_task8/iter_140000.pth"

resume_from = None
detect_task = dict(full_body=0)
extra_task = ['track_id']

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
conv_module = dict(type='ConvModule', kernel_size=3, padding=1, bias=False, norm_cfg=dict(type='MMSyncBN'))
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
    cls_head=dict(
        type='FMHead',
        gt_jitter=0.2,
        in_channels=32,
        feat_channels=32,
        feat_out_channels=32,
        reid_num=2125,
        feature_norm=False,
        fc_bias=True,
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
                loss=dict(type='CrossEntropyLoss', loss_weight=0.05)),
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
    det_score_thrd=[0.3, 0.3],  # detection threshold for predict box
    cls_score_thrd=[0.5, 0.5],  # cls threshold for predict box
    rel_cls_score_thrd=0.5,  # cls threshold for relation pair cls
    hoi_rel_thrd=1e-2,  # hoi_relation threshold for predict box
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=100)

dataset_type = 'CustomTaskDataset'
mot_root = '/mnt/lustre/share_data/parrots_model_data/objectflow_mot/motpkl/mot/'
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
    dict(type='Resize', img_scale=(960, 544), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),  # recommend
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='DefaultFormatBundleTask'),
    dict(
        type='CollectTask',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'], output_extra_anns=True,
        extra_keys=extra_task)
]
val_pipeline = [
    dict(type='LoadImageFromFile', color_type='color', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 544),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='DefaultFormatBundleTask'),
            dict(type='CollectTask', keys=['img','gt_labels', 'gt_bboxes'],
                 output_extra_anns=True, extra_keys=extra_task),
        ])
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
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'CustomTaskDataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=[
            # mot_root + '2dmot2015/pkl/train.pkl',
            # mot_root + 'mot16/pkl/train.pkl',
            # mot_root + 'mot20/pkl/train.pkl',
            # mot_root + 'emb600id/pkl/train.pkl',
            # mot_root + 'prw/pkl/train.pkl',
            mot_root + 'HKJC/2.pkl',
        ],
        img_prefix=[
            # 'sh1985:s3://mot/mnt/lustre/share/lindelv/data/mot/',
            # 'sh1985:s3://mot/mnt/lustre/share/lindelv/data/mot/',
            # 'sh1985:s3://mot/mnt/lustre/share/lindelv/data/mot/',
            # 'sh1985:s3://mot/mnt/lustre/share/lindelv/data/mot/',
            # 'sh1985:s3://mot/mnt/lustre/share/lindelv/data/mot/',
            'sh1984:s3://parrots_model_data/objectflow_mot/mot/mnt/lustre/datatag/',
        ],
        pipeline=train_pipeline,
        with_ignore_bboxes=True,
        extra_task=extra_task,
        detect_task=detect_task),
    val=[
        dict(
            type=dataset_type,
            ann_file=mot_root + '2dmot2015/pkl/train.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/mot/mnt/lustre/share/lindelv/data/mot/',
            pipeline=val_pipeline,
            test_mode=False,
            with_ignore_bboxes=True,
            detect_task=detect_task,
            extra_task=extra_task,
            test_cfg=test_cfg),
    ],
    test=[
        dict(
            type=dataset_type,
            ann_file=mot_root + '2dmot2015/pkl/train.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/mot/mnt/lustre/share/lindelv/data/mot/',
            pipeline=test_pipeline,
            with_ignore_bboxes=True,
            test_mode=True,
            detect_task=detect_task,
            extra_task=extra_task,
            test_cfg=test_cfg),
    ],
)
