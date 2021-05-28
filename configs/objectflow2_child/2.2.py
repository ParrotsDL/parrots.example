epoch_size = 10000
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[100000, 200000])
evaluation = dict(
    interval=300000,
    metric=dict(bbox_head=dict(metric='RatP', target_prec=0.99)))
max_iters = 250000
workflow = [('train', 300000)]
# task_groups = dict(
#     face=[0, 1, 2, 3, 4, 5, 6, 7],
#     body=[8, 9, 10, 11, 12, 13, 14, 15],
#     facebody=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#     child=[16, 17, 18, 19, 20, 21, 22, 23],
#     relation=[24, 25, 26, 27, 28, 29, 30, 31])
task_groups = dict(
    face=[0, 1, 2, 3],
    body=[4, 5, 6, 7],
    facebody=[0, 1, 2, 3, 4, 5, 6, 7],
    child=[8, 9, 10, 11],
    relation=[12,13, 14, 15])
task_prefix = dict({
    'neck': 'facebody',
    'bbox_head.cls_convs': 'facebody',
    'bbox_head.reg_convs': 'facebody',
    'bbox_head.retina_reg': 'facebody',
    'bbox_head.retina_cls': 'body'
})
checkpoint_config = dict(interval=5000)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/mnt/lustre/share_data/parrots_model_ckpt/objectflow_child/multitask_base/body.pth'
resume_from = None
model = dict(
    type='RetinaNetMhMtMFPN',
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
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
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
dataset_type = 'CustomTaskDataset'
# data_root_pmdb_train = '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_v0.1.0/'
# data_root_pmdb_test = '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/'
# child_data_root_pmdb_train = '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/'
# child_data_root_pmdb_test = '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_test_v0.1.0/'
detect_task = dict(half_body=0)
extra_task = []
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='color',
        file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='MixBadcase',
    #     ratio=0.25,
    #     mode='void_gt',
    #     ann_file='/mnt/lustre/fangkairen/data/child_dataset/dollv2_all.pkl',
    #     img_prefix='/mnt/lustre/fangkairen/data/child_dataset/dollv2_all/'),
    # dict(
    #     type='MixBadcase',
    #     ratio=0.25,
    #     mode='void_gt',
    #     ann_file='/mnt/lustre/fangkairen/data/child_dataset/shubaov2_all.pkl',
    #     img_prefix='/mnt/lustre/fangkairen/data/child_dataset/shubaov2_all/'),
    dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
    dict(type='Resize', img_scale=(640, 384), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectTask',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'],
        extra_keys=[])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='color',
        file_client_args=dict(backend='petrel')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_datasets = [
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_0413_test.pmdb',
    #     img_prefix=
    #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/new_child_0413/child_test/rotated_child_1920/',
    #     with_ignore_bboxes=True,
    #     detect_task=dict(half_body=0),
    #     extra_task=[],
    #     pipeline=[
    #         dict(
    #             type='LoadImageFromFile',
    #             color_type='color',
    #             file_client_args=dict(backend='petrel')),
    #         dict(
    #             type='MultiScaleFlipAug',
    #             img_scale=(640, 384),
    #             flip=False,
    #             transforms=[
    #                 dict(type='Resize', keep_ratio=True),
    #                 dict(type='RandomFlip'),
    #                 dict(
    #                     type='Normalize',
    #                     mean=[0.0, 0.0, 0.0],
    #                     std=[1.0, 1.0, 1.0],
    #                     to_rgb=False),
    #                 dict(type='Pad', size_divisor=32),
    #                 dict(type='ImageToTensor', keys=['img']),
    #                 dict(type='Collect', keys=['img'])
    #             ])
    #     ]),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_0508_test.pmdb',
    #     img_prefix=
    #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/child_0508/child_test/rotated_child_0508/',
    #     with_ignore_bboxes=True,
    #     detect_task=dict(half_body=0),
    #     extra_task=[],
    #     pipeline=[
    #         dict(
    #             type='LoadImageFromFile',
    #             color_type='color',
    #             file_client_args=dict(backend='petrel')),
    #         dict(
    #             type='MultiScaleFlipAug',
    #             img_scale=(640, 384),
    #             flip=False,
    #             transforms=[
    #                 dict(type='Resize', keep_ratio=True),
    #                 dict(type='RandomFlip'),
    #                 dict(
    #                     type='Normalize',
    #                     mean=[0.0, 0.0, 0.0],
    #                     std=[1.0, 1.0, 1.0],
    #                     to_rgb=False),
    #                 dict(type='Pad', size_divisor=32),
    #                 dict(type='ImageToTensor', keys=['img']),
    #                 dict(type='Collect', keys=['img'])
    #             ])
    #     ]),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_483.pmdb',
    #     img_prefix=
    #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/child_483/child_test/rotated_child_483/',
    #     with_ignore_bboxes=True,
    #     detect_task=dict(half_body=0),
    #     extra_task=[],
    #     pipeline=[
    #         dict(
    #             type='LoadImageFromFile',
    #             color_type='color',
    #             file_client_args=dict(backend='petrel')),
    #         dict(
    #             type='MultiScaleFlipAug',
    #             img_scale=(640, 384),
    #             flip=False,
    #             transforms=[
    #                 dict(type='Resize', keep_ratio=True),
    #                 dict(type='RandomFlip'),
    #                 dict(
    #                     type='Normalize',
    #                     mean=[0.0, 0.0, 0.0],
    #                     std=[1.0, 1.0, 1.0],
    #                     to_rgb=False),
    #                 dict(type='Pad', size_divisor=32),
    #                 dict(type='ImageToTensor', keys=['img']),
    #                 dict(type='Collect', keys=['img'])
    #             ])
    #     ]),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12432.pmdb',
    #     img_prefix=
    #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12432/',
    #     with_ignore_bboxes=True,
    #     detect_task=dict(half_body=0),
    #     extra_task=[],
    #     pipeline=[
    #         dict(
    #             type='LoadImageFromFile',
    #             color_type='color',
    #             file_client_args=dict(backend='petrel')),
    #         dict(
    #             type='MultiScaleFlipAug',
    #             img_scale=(640, 384),
    #             flip=False,
    #             transforms=[
    #                 dict(type='Resize', keep_ratio=True),
    #                 dict(type='RandomFlip'),
    #                 dict(
    #                     type='Normalize',
    #                     mean=[0.0, 0.0, 0.0],
    #                     std=[1.0, 1.0, 1.0],
    #                     to_rgb=False),
    #                 dict(type='Pad', size_divisor=32),
    #                 dict(type='ImageToTensor', keys=['img']),
    #                 dict(type='Collect', keys=['img'])
    #             ])
    #     ]),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12433.pmdb',
    #     img_prefix=
    #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12433/',
    #     with_ignore_bboxes=True,
    #     detect_task=dict(half_body=0),
    #     extra_task=[],
    #     pipeline=[
    #         dict(
    #             type='LoadImageFromFile',
    #             color_type='color',
    #             file_client_args=dict(backend='petrel')),
    #         dict(
    #             type='MultiScaleFlipAug',
    #             img_scale=(640, 384),
    #             flip=False,
    #             transforms=[
    #                 dict(type='Resize', keep_ratio=True),
    #                 dict(type='RandomFlip'),
    #                 dict(
    #                     type='Normalize',
    #                     mean=[0.0, 0.0, 0.0],
    #                     std=[1.0, 1.0, 1.0],
    #                     to_rgb=False),
    #                 dict(type='Pad', size_divisor=32),
    #                 dict(type='ImageToTensor', keys=['img']),
    #                 dict(type='Collect', keys=['img'])
    #             ])
    #     ]),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12450.pmdb',
    #     img_prefix=
    #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12450/',
    #     with_ignore_bboxes=True,
    #     detect_task=dict(half_body=0),
    #     extra_task=[],
    #     pipeline=[
    #         dict(
    #             type='LoadImageFromFile',
    #             color_type='color',
    #             file_client_args=dict(backend='petrel')),
    #         dict(
    #             type='MultiScaleFlipAug',
    #             img_scale=(640, 384),
    #             flip=False,
    #             transforms=[
    #                 dict(type='Resize', keep_ratio=True),
    #                 dict(type='RandomFlip'),
    #                 dict(
    #                     type='Normalize',
    #                     mean=[0.0, 0.0, 0.0],
    #                     std=[1.0, 1.0, 1.0],
    #                     to_rgb=False),
    #                 dict(type='Pad', size_divisor=32),
    #                 dict(type='ImageToTensor', keys=['img']),
    #                 dict(type='Collect', keys=['img'])
    #             ])
    #     ]),
    dict(
        type='CustomTaskDataset',
        ann_file=
        '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_test_v0.1.0/child_test_20200927.pmdb',
        img_prefix='s3://parrots_model_data/objectflow_child/VACabin/datasets/',
        with_ignore_bboxes=True,
        detect_task=dict(half_body=0),
        extra_task=[],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color',
                file_client_args=dict(backend='petrel')),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CustomTaskDataset',
        ann_file=[
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20201213.pmdb',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations/wanou_20210116.pkl',
        ],
        img_prefix=[
            's3://parrots_model_data/objectflow_child/VACabin/datasets/child_cabin_oms_20201213/OMS/',
            's3://parrots_model_data/objectflow_child/VACabin/wannou_20210116/',
        ],
        with_ignore_bboxes=True,
        detect_task=dict(half_body=0),
        extra_task=[],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color',
                file_client_args=dict(backend='petrel')),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(
            #     type='MixBadcase',
            #     ratio=0.25,
            #     mode='void_gt',
            #     ann_file=
            #     '/mnt/lustre/fangkairen/data/child_dataset/dollv2_all.pkl',
            #     img_prefix=
            #     '/mnt/lustre/fangkairen/data/child_dataset/doll_png/'),
            # dict(
            #     type='MixBadcase',
            #     ratio=0.25,
            #     mode='void_gt',
            #     ann_file=
            #     '/mnt/lustre/fangkairen/data/child_dataset/shubaov2_all.pkl',
            #     img_prefix=
            #     '/mnt/lustre/fangkairen/data/child_dataset/bag_png/'),
            # dict(
            #     type='MixBadcase',
            #     ratio=0.25,
            #     mode='void_gt',
            #     ann_file=
            #     '/mnt/lustre/datatag/fangkairen/child_dataset/zhangao_child_attack.pkl',
            #     img_prefix=
            #     '/mnt/lustre/datatag/fangkairen/child_dataset/zhangao_child_attack/'),
            dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
            dict(type='Resize', img_scale=(640, 384), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='CollectTask',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'],
                extra_keys=[])
        ]),
    val=[
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_0413_test.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/new_child_0413/child_test/rotated_child_1920/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_0508_test.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/child_0508/child_test/rotated_child_0508/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_483.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/child_483/child_test/rotated_child_483/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12432.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12432/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12433.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12433/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12450.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12450/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        dict(
            type='CustomTaskDataset',
            ann_file=
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_test_v0.1.0/child_test_20200927.pmdb',
            img_prefix='s3://parrots_model_data/objectflow_child/VACabin/datasets/',
            with_ignore_bboxes=True,
            detect_task=dict(half_body=0),
            extra_task=[],
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    color_type='color',
                    file_client_args=dict(backend='petrel')),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 384),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(
                            type='Normalize',
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            to_rgb=False),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ])
    ],
    test=[
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_0413_test.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/new_child_0413/child_test/rotated_child_1920/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_0508_test.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/child_0508/child_test/rotated_child_0508/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/child_483.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/dataset_chengwenhao/child_483/child_test/rotated_child_483/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12432.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12432/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12433.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12433/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/body_det_test_v0.1.0/reanno_12450.pmdb',
        #     img_prefix=
        #     'sh1984:s3://detection_dataset_zqm/body_v0.1.0/mnt/lustrenew/dutianyuan/detection_data/body_data/reanno_dataset/reanno_12450/',
        #     with_ignore_bboxes=True,
        #     detect_task=dict(half_body=0),
        #     extra_task=[],
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(type='RandomFlip'),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='ImageToTensor', keys=['img']),
        #                 dict(type='Collect', keys=['img'])
        #             ])
        #     ]),
        dict(
            type='CustomTaskDataset',
            ann_file=
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_test_v0.1.0/child_test_20200927.pmdb',
            img_prefix='s3://parrots_model_data/objectflow_child/VACabin/datasets/',
            with_ignore_bboxes=True,
            detect_task=dict(half_body=0),
            extra_task=[],
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    color_type='color',
                    file_client_args=dict(backend='petrel')),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 384),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(
                            type='Normalize',
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            to_rgb=False),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ])
    ])
work_dir = './workdir/multitask_void_bag_doll_1116_0.2_new/_task1'
gpu_ids = range(0, 1)
