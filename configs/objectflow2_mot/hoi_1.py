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
    interval=epoch_size,
    metric=dict(
        relation_head=dict(
            metric='RelationTest',
            target_prec=0.95,
            GT_MODE=True,
            start_task='face_cls',
            end_task='body_cls',
            det_index=dict(face_cls=0, body_cls=1))))

max_iters = epoch_size * 30
workflow = [('train', max_iters)]
# fix_prefix = ['backbone', 'neck', 'cls_head', 'bbox_head']
fix_prefix = []
# task_groups = {'hoi': [i for i in range(0, 8)]}
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
task_prefix = {'neck.lateral_convs.0': 'hoi',
               'neck.lateral_convs.1': 'hoi',
               'neck.lateral_convs.2': 'hoi',
               'neck.d_conv': 'hoi',
               'neck.fpn_convs.0': 'hoi',
               'neck.fpn_convs.1': None,
               'neck.fpn_convs.2': None,
               'neck.fpn_convs.3': None,
               'neck.fpn_convs.4': None,
               'relation_head': 'hoi'}

checkpoint_config = dict(interval=epoch_size)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "/mnt/lustre/share_data/parrots_model_ckpt/objectflow_mot/facebodyhoireid_20_3/_task7/iter_140000.pth"
resume_from = None

detect_task = dict(face=0, full_body=1)
extra_task = ['track_id', 'face_cls', 'body_cls']
extra_detect_map = dict(face_cls='face', body_cls='full_body')

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
    relation_head=dict(
        type='HOIHeadRelation',
        rel_radius=3,
        norm_scale=2,
        single_avg=False,
        in_channels=32,
        feat_channels=32,
        num_classes_object=1,
        num_classes_verb=2,
        subject_labels=[0],
        feature_stride=8,
        feature_level=0,
        max_objs=128,
        max_rels=64,
        relation_key=None,
        reg_center_offset=False,
        with_bbox_head=False,
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1),
        loss_reg=dict(type='SmoothL1Loss', reduction='mean', loss_weight=1)),
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

mot_root = '/mnt/lustre/share_data/parrots_model_data/objectflow_mot/motpkl/'
dataset_type = 'CustomTaskDataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=[
            mot_root + 'face_detection/new_annos/task_14484.pkl',
            # mot_root + 'face_detection/new_annos/task_14411.pkl',
            mot_root + 'mot/crowdhuman/pkl/crowdhuman_train_le50.pkl',
            # mot_root + 'mot/crowdhuman/pkl/crowdhuman_val_le50.pkl',
            mot_root + 'mot/HKJCV1testfilter_20201026.pkl',
            mot_root + 'mot/HKJCV2testfilter_20201029.pkl',
        ],
        img_prefix=[
            'sh1984:s3://parrots_model_data/objectflow_mot/wangliming/data/detection_data/shangfei_mendian/task_14484/Images/',
            # '/mnt/lustre/share/zhangqiming/wangliming/data//detection_data/task_14411/Images/',
            'sh1984:s3://parrots_model_data/objectflow_mot/wangliming/data/detection_data/CrowdHuman/CrowdHuman_train/Images/',
            # '/mnt/lustre/share/zhangqiming/wangliming/data//detection_data/CrowdHuman//CrowdHuman_val/Images/',
            'sh1984:s3://parrots_model_data/objectflow_mot/data/mot/pack_HKJCV1/frames/',
            'sh1984:s3://parrots_model_data/objectflow_mot/data/horce_race_pan/',
        ],
        pipeline=train_pipeline,
        with_ignore_bboxes=True,
        extra_task=extra_task,
        detect_task=detect_task),
    val=[
        dict(
            type=dataset_type,
            ann_file=mot_root + 'mot/HKJCV1testfilter_20201026.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/data/mot/pack_HKJCV1/frames/',
            pipeline=val_pipeline,
            test_mode=False,
            with_ignore_bboxes=True,
            detect_task=detect_task,
            extra_task=extra_task,
            test_cfg=test_cfg),
        dict(
            type=dataset_type,
            ann_file=mot_root + 'mot/HKJCV2testfilter_20201029.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/data/horce_race_pan/',
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
            ann_file=mot_root + 'mot/HKJCV1testfilter_20201026.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/data/mot/pack_HKJCV1/frames/',
            pipeline=val_pipeline,
            with_ignore_bboxes=True,
            test_mode=False,
            detect_task=detect_task,
            extra_task=extra_task,
            test_cfg=test_cfg),
        dict(
            type=dataset_type,
            ann_file=mot_root + 'mot/HKJCV2testfilter_20201029.pkl',
            img_prefix='sh1984:s3://parrots_model_data/objectflow_mot/data/horce_race_pan/',
            pipeline=val_pipeline,
            test_mode=False,
            with_ignore_bboxes=True,
            detect_task=detect_task,
            extra_task=extra_task,
            test_cfg=test_cfg),
    ],
)
