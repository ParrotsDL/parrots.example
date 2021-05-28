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
    metric=dict(
        cls_head=dict(metric='ClsTest', target_prec=0.95, GT_MODE=True)))
max_iters = 250000
workflow = [('train', 300000)]
fix_prefix = []
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
    'cls_neck.lateral_convs.0': 'child',
    'cls_neck.lateral_convs.1': 'child',
    'cls_neck.lateral_convs.2': 'child',
    'cls_neck.d_conv': 'child',
    'cls_neck.fpn_convs.0': 'child',
    'cls_neck.fpn_convs.1': None,
    'cls_neck.fpn_convs.2': None,
    'cls_neck.fpn_convs.3': None,
    'cls_neck.fpn_convs.4': None,
    'cls_head': 'child'
})
checkpoint_config = dict(interval=5000)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/mnt/lustre/share_data/parrots_model_ckpt/objectflow_child/multitask_base/recog.pth'
resume_from = None
detect_task = dict(face=0, half_body=1)
extra_task = ['track_id', 'face_cls', 'body_cls']
extra_detect_map = dict(face_cls='face', body_cls='half_body')
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
conv_module = dict(
    type='ConvModule',
    kernel_size=3,
    padding=1,
    bias=False,
    norm_cfg=dict(type='MMSyncBN'))
model = dict(
    type='RetinaNetMhMtMFPN',
    backbone=dict(
        type='MobileNetV2_ImgNet',
        last_feat_channel=160,
        img_channel=3,
        out_indices=(0, 1, 2, 3),
        normalize=dict(type='MMSyncBN')),
    cls_neck=dict(
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
        feat_channels=16,
        feat_out_channels=16,
        aio_refine_share=True,
        shared_convs_cfg=dict(
            type='ChooseOneFromListInTensorOut',
            choose_ind=0,
            conv_list=[
                dict(
                    type='ConvModule',
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='MMSyncBN')),
                dict(
                    type='ConvModule',
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='MMSyncBN')),
                dict(
                    type='ConvModule',
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm_cfg=dict(type='MMSyncBN'))
            ]),
        branch_cfg_list=[
            dict(
                name='face_cls',
                base_task='face',
                type='aiocls',
                net=dict(
                    type='TensorInListOutRefine',
                    conv_list=[
                        dict(type='ConvModule', kernel_size=1),
                        dict(type='ConvModule', kernel_size=1, act_cfg=None)
                    ]),
                cls_channels=2,
                loss=dict(type='MSELoss', loss_weight=1)),
            dict(
                name='body_cls',
                base_task='half_body',
                type='aiocls',
                net=dict(
                    type='TensorInListOutRefine',
                    conv_list=[
                        dict(type='ConvModule', kernel_size=1),
                        dict(type='ConvModule', kernel_size=1, act_cfg=None)
                    ]),
                cls_channels=2,
                loss=dict(type='MSELoss', loss_weight=1))
        ]))
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
    det_score_thrd=[0.3, 0.3],
    cls_score_thrd=[0.5, 0.5],
    rel_cls_score_thrd=0.5,
    hoi_rel_thrd=0.01,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=100)
dataset_type = 'CustomTaskDataset'
# data_root_pmdb_train = '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/'
# data_root_pmdb_test = '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_test_v0.1.0/'
# gaosi_root = '/mnt/lustre/share_data/zhangao/VACabin/child_train_test/gaosi_room/'
# root_du = '/mnt/lustrenew/dutianyuan/detection_data/'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='color',
        file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
    dict(type='Resize', img_scale=(640, 384), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='DefaultFormatBundleTask'),
    dict(
        type='CollectTask',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'],
        output_extra_anns=True,
        extra_keys=['track_id', 'face_cls', 'body_cls'])
]
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        color_type='color',
        file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='DefaultFormatBundleTask'),
            dict(
                type='CollectTask',
                keys=['img', 'gt_labels', 'gt_bboxes'],
                output_extra_anns=True,
                extra_keys=['track_id', 'face_cls', 'body_cls'])
        ])
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
val_datasets = [
    dict(
        type='CustomTaskDataset',
        ann_file=
        '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_test_v0.1.0/child_test_20200927.pmdb',
        img_prefix='s3://parrots_model_data/objectflow_child/VACabin/datasets/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color',
                file_client_args=dict(backend='petrel')),
            dict(type='LoadAnnotationsInputList', with_bbox=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='DefaultFormatBundleTask'),
                    dict(
                        type='CollectTask',
                        keys=['img', 'gt_labels', 'gt_bboxes'],
                        output_extra_anns=True,
                        extra_keys=['track_id', 'face_cls', 'body_cls'])
                ])
        ],
        test_mode=False,
        with_ignore_bboxes=True,
        detect_task=dict(face=0, half_body=1),
        extra_task=['track_id', 'face_cls', 'body_cls']),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_test_v0.2.0/child_test_20201122.pmdb',
    #     img_prefix='sh1984:s3://VACabin/datasets/',
    #     pipeline=[
    #         dict(
    #             type='LoadImageFromFile',
    #             color_type='color',
    #             file_client_args=dict(backend='petrel')),
    #         dict(type='LoadAnnotationsInputList', with_bbox=True),
    #         dict(
    #             type='MultiScaleFlipAug',
    #             img_scale=(640, 384),
    #             flip=False,
    #             transforms=[
    #                 dict(type='Resize', keep_ratio=True),
    #                 dict(
    #                     type='Normalize',
    #                     mean=[0.0, 0.0, 0.0],
    #                     std=[1.0, 1.0, 1.0],
    #                     to_rgb=False),
    #                 dict(type='Pad', size_divisor=32),
    #                 dict(type='DefaultFormatBundle'),
    #                 dict(type='DefaultFormatBundleTask'),
    #                 dict(
    #                     type='CollectTask',
    #                     keys=['img', 'gt_labels', 'gt_bboxes'],
    #                     output_extra_anns=True,
    #                     extra_keys=['track_id', 'face_cls', 'body_cls'])
    #             ])
    #     ],
    #     test_mode=False,
    #     with_ignore_bboxes=True,
    #     detect_task=dict(face=0, half_body=1),
    #     extra_task=['track_id', 'face_cls', 'body_cls'])
]
test_datasets = [
    dict(
        type='CustomTaskDataset',
        ann_file=
        '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_test_v0.1.0/child_test_20200927.pmdb',
        img_prefix='s3://parrots_model_data/objectflow_child/VACabin/datasets/',
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
        ],
        with_ignore_bboxes=True,
        test_mode=True,
        detect_task=dict(face=0, half_body=1),
        extra_task=['track_id', 'face_cls', 'body_cls']),
    # dict(
    #     type='CustomTaskDataset',
    #     ann_file=
    #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_test_v0.2.0/child_test_20201122.pmdb',
    #     img_prefix='sh1984:s3://VACabin/datasets/',
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
    #     ],
    #     with_ignore_bboxes=True,
    #     test_mode=True,
    #     detect_task=dict(face=0, half_body=1),
    #     extra_task=['track_id', 'face_cls', 'body_cls'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='CustomTaskDataset',
        ann_file=[
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_v0.2.0/child_indoor_outdoor_ipc_oms_phone_20200909.pmdb',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_v0.2.0/child_indoor_outdoor_ipc_oms_phone_20200910.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20200909.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20200910.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20200911.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20200923.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20200925.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20200927.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20200914.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_indoor_outdoor_ipc_oms_phone_20200913.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_indoor_outdoor_ipc_oms_phone_20200924.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_indoor_outdoor_ipc_oms_phone_20200915.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_indoor_outdoor_ipc_oms_phone_20200925.pmdb',

            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20201109.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20201110.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_cabin_oms_20201114.pmdb',
            # '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_v0.2.0/child_indoor_outdoor_ipc_oms_phone_20200928.pmdb',
        ],
        img_prefix=[
            's3://parrots_model_data/objectflow_child/VACabin/datasets/child_indoor_outdoor_ipc_oms_phone_20200909',
            's3://parrots_model_data/objectflow_child/VACabin/datasets/child_indoor_outdoor_ipc_oms_phone_20200910',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20200909',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20200910',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20200911',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20200923',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20200925',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20200927',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20200914',
            # 'sh1984:s3://VACabin/datasets/child_cabin_indoor_outdoor_ipc_oms_phone_20200913',
            # 'sh1984:s3://VACabin/datasets/child_indoor_outdoor_ipc_oms_phone_20200924',
            # 'sh1984:s3://VACabin/datasets/child_indoor_outdoor_ipc_oms_phone_20200915',
            # 'sh1984:s3://VACabin/datasets/child_indoor_outdoor_ipc_oms_phone_20200925',
            
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20201109',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20201110',
            # 'sh1984:s3://VACabin/datasets/child_cabin_oms_20201114',
            # 'sh1984:s3://VACabin/datasets/child_indoor_outdoor_ipc_oms_phone_20200928',
        ],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color',
                file_client_args=dict(backend='petrel')),
            dict(type='LoadAnnotationsInputList', with_bbox=True),
            dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
            dict(type='Resize', img_scale=(640, 384), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='DefaultFormatBundleTask'),
            dict(
                type='CollectTask',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'],
                output_extra_anns=True,
                extra_keys=['track_id', 'face_cls', 'body_cls'])
        ],
        with_ignore_bboxes=True,
        extra_task=['track_id', 'face_cls', 'body_cls'],
        detect_task=dict(face=0, half_body=1)),
    val=[
        dict(
            type='CustomTaskDataset',
            ann_file=
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_test_v0.1.0/child_test_20200927.pmdb',
            img_prefix='s3://parrots_model_data/objectflow_child/VACabin/datasets/',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    color_type='color',
                    file_client_args=dict(backend='petrel')),
                dict(type='LoadAnnotationsInputList', with_bbox=True),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 384),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(
                            type='Normalize',
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            to_rgb=False),
                        dict(type='Pad', size_divisor=32),
                        dict(type='DefaultFormatBundle'),
                        dict(type='DefaultFormatBundleTask'),
                        dict(
                            type='CollectTask',
                            keys=['img', 'gt_labels', 'gt_bboxes'],
                            output_extra_anns=True,
                            extra_keys=['track_id', 'face_cls', 'body_cls'])
                    ])
            ],
            test_mode=False,
            with_ignore_bboxes=True,
            detect_task=dict(face=0, half_body=1),
            extra_task=['track_id', 'face_cls', 'body_cls']),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_test_v0.2.0/child_test_20201122.pmdb',
        #     img_prefix='sh1984:s3://VACabin/datasets/',
        #     pipeline=[
        #         dict(
        #             type='LoadImageFromFile',
        #             color_type='color',
        #             file_client_args=dict(backend='petrel')),
        #         dict(type='LoadAnnotationsInputList', with_bbox=True),
        #         dict(
        #             type='MultiScaleFlipAug',
        #             img_scale=(640, 384),
        #             flip=False,
        #             transforms=[
        #                 dict(type='Resize', keep_ratio=True),
        #                 dict(
        #                     type='Normalize',
        #                     mean=[0.0, 0.0, 0.0],
        #                     std=[1.0, 1.0, 1.0],
        #                     to_rgb=False),
        #                 dict(type='Pad', size_divisor=32),
        #                 dict(type='DefaultFormatBundle'),
        #                 dict(type='DefaultFormatBundleTask'),
        #                 dict(
        #                     type='CollectTask',
        #                     keys=['img', 'gt_labels', 'gt_bboxes'],
        #                     output_extra_anns=True,
        #                     extra_keys=['track_id', 'face_cls', 'body_cls'])
        #             ])
        #     ],
        #     test_mode=False,
        #     with_ignore_bboxes=True,
        #     detect_task=dict(face=0, half_body=1),
        #     extra_task=['track_id', 'face_cls', 'body_cls'])
    ],
    test=[
        dict(
            type='CustomTaskDataset',
            ann_file=
            '/mnt/lustre/share_data/parrots_model_data/objectflow_child/VACabin/annotations_pmdb/child_recog_test_v0.1.0/child_test_20200927.pmdb',
            img_prefix='s3://parrots_model_data/objectflow_child/VACabin/datasets/',
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
            ],
            with_ignore_bboxes=True,
            test_mode=True,
            detect_task=dict(face=0, half_body=1),
            extra_task=['track_id', 'face_cls', 'body_cls']),
        # dict(
        #     type='CustomTaskDataset',
        #     ann_file=
        #     '/mnt/lustre/share_data/zhangao/VACabin/annotations_pmdb/child_recog_test_v0.2.0/child_test_20201122.pmdb',
        #     img_prefix='sh1984:s3://VACabin/datasets/',
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
        #     ],
        #     with_ignore_bboxes=True,
        #     test_mode=True,
        #     detect_task=dict(face=0, half_body=1),
        #     extra_task=['track_id', 'face_cls', 'body_cls'])
    ])
work_dir = './workdir/multitask_void_bag_doll_1116_0.2_new/_task2'
gpu_ids = range(0, 1)
