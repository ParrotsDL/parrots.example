epoch_size = 5423
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16*epoch_size, 22*epoch_size])
evaluation = dict(interval=epoch_size,
                  metric=dict(bbox_head=dict(metric='RatP', target_prec=0.99)))
max_iters = epoch_size * 24
workflow = [('train', max_iters)]

task_groups = {'cls': [i for i in range(0, 8)],
               'reg': [i for i in range(8,16)]}
task_prefix = {'bbox_head.retina_cls': 'cls',
               'bbox_head.retina_reg': 'cls'}

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
    pretrained='/mnt/lustre/share_data/parrots_model_ckpt/objectflow_object/mb2_1x_gray_top1_71.18_top5_89.93.pth.tar',
    backbone=dict(
        type='MobileNetV2_ImgNet',
        last_feat_channel=160,
        img_channel=1,
        norm_eval=False,
        out_indices=(0, 1, 2, 3),
        normalize=dict(type='MMSyncBN')),
    neck=dict(
        type='DCONV_FPN',
        in_channels=[24, 32, 96, 160],
        out_channels=32,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='MMSyncBN'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=32,
        stacked_convs=4,
        feat_channels=32,
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
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.0)))
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
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.3),
    max_per_img=100)
# dataset settings
dataset_type = 'CustomTaskDataset'
data_root_pmdb_train = '/mnt/lustre/share_data/parrots_model_data/objectflow_object/object_pmdb/train/'
data_root_pmdb_test = '/mnt/lustre/share_data/parrots_model_data/objectflow_object/object_pmdb/val/'

detect_task = {"phone": 0, "cup":1}
extra_task = []

img_norm_cfg = dict(mean=[0.0], std=[1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotationsInputList', with_bbox=True),
    dict(
        type='GrayPhotoMetricDistortion',
        brightness_delta=5,
        contrast_range=(0.95, 1.05),
        saturation_range=(0.95, 1.05),
        hue_delta=5,
        prob=0.6),
    dict(type='Gaussian_noise', sigma=0.2, prob=0.4),
    dict(type='RandomOcclution_plus', ratio=0.5, max_iou=0.1),
    dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
    dict(type='Resize', img_scale=(640, 384), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='DefaultFormatBundleTask'),
    dict(
        type='CollectTask',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'],
        extra_keys=[])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', file_client_args=dict(backend='petrel')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 384),
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
test_datasets = [
    dict(
        type=dataset_type,
        ann_file=data_root_pmdb_test + 'commonObjectOMSTest_20200901.pmdb',
        img_prefix='/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/rotated_object/',
        with_ignore_bboxes=True,  # gt_bboxes_ignore
        detect_task=detect_task,  # start from 0
        extra_task=extra_task,
        pipeline=test_pipeline),
    dict(
        type=dataset_type,
        ann_file=data_root_pmdb_test + 'commonObjectShoujiTest_20200901.pmdb',
        img_prefix='/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/rotated_object/',
        with_ignore_bboxes=True,  # gt_bboxes_ignore
        detect_task=detect_task,  # start from 0
        extra_task=extra_task,
        pipeline=test_pipeline),
    # dict(
    #     type=dataset_type,
    #     ann_file=data_root_pmdb_test + 'objectRemove300IRTest_20201008.pmdb',
    #     img_prefix='sh41:s3://zhangao/rotated_object/',
    #     with_ignore_bboxes=True,  # gt_bboxes_ignore
    #     detect_task=detect_task,  # start from 0
    #     extra_task=extra_task,
    #     pipeline=test_pipeline),
    # dict(
    #     type=dataset_type,
    #     ann_file=data_root_pmdb_test + 'objectRemove300RGBTest_20201008.pmdb',
    #     img_prefix='sh41:s3://zhangao/rotated_object/',
    #     with_ignore_bboxes=True,  # gt_bboxes_ignore
    #     detect_task=detect_task,  # start from 0
    #     extra_task=extra_task,
    #     pipeline=test_pipeline),
    # dict(
    #     type=dataset_type,
    #     ann_file=data_root_pmdb_test + '20190914_0916_all_new_pure_val_crop.pmdb',
    #     img_prefix='sh1984:s3://detection_dataset_zqm/phone_v0.1.0/mnt/lustre/zhangao/dataset_chengwenhao/detection_data/phone_data/p2/phone2/bmw_collection/20190914_0916_all_new/rotated_images_crop/',
    #     with_ignore_bboxes=True,  # gt_bboxes_ignore
    #     detect_task=detect_task,  # start from 0
    #     extra_task=extra_task,
    #     pipeline=test_pipeline),
    # dict(
    #     type=dataset_type,
    #     ann_file=data_root_pmdb_test + 'objectOmsTest_20201029.pmdb',
    #     img_prefix='sh41:s3://zhangao/object_oms_image/',
    #     with_ignore_bboxes=True,  # gt_bboxes_ignore
    #     detect_task=detect_task,  # start from 0
    #     extra_task=extra_task,
    #     pipeline=test_pipeline),
    # dict(
    #     type=dataset_type,
    #     ann_file=data_root_pmdb_test + 'objectOmsChildTest_20201029.pmdb',
    #     img_prefix='sh41:s3://zhangao/object_oms_image/',
    #     with_ignore_bboxes=True,  # gt_bboxes_ignore
    #     detect_task=detect_task,  # start from 0
    #     extra_task=extra_task,
    #     pipeline=test_pipeline),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root_pmdb_train + 'dms_20200716_negsmok.pmdb',
            data_root_pmdb_train + 'carIndoorOutdoor_20200716_action.pmdb',
            data_root_pmdb_train + 'ylwt_yes.pmdb',
            data_root_pmdb_train + 'all_phone_newdataset.pmdb',
            data_root_pmdb_train + '20190914_0916_all_new_train_crop.pmdb',
            data_root_pmdb_train + 'commonObjectTrainAll_20200901.pmdb',
            data_root_pmdb_train + 'commonObjectTrainAllpos_20200901.pmdb',
            # data_root_pmdb_train + 'objectRemove300Train_20201008.pmdb',
            # data_root_pmdb_train + 'objectRemove300Trainpos_20201008.pmdb',
            # data_root_pmdb_train + 'objectRemove300Trainpos_20201008.pmdb',
            # data_root_pmdb_train + 'objects365_train_new_label_remove_cup_phone_5w.pmdb',
            # data_root_pmdb_train + 'objectproRemoveIgnoreRemovefreecallTrain_20200918.pmdb',
            # data_root_pmdb_train + 'objectOmsTrain_20201120.pmdb',
        ],
        img_prefix=[
            '/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/hoi_dataset/',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/hoi_dataset/',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/hoi_dataset/rotated_ylwt/',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/hoi_dataset/rotated_all_phone/',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/hoi_dataset/rotated_images_crop/',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/rotated_object/',
            '/mnt/lustre/share_data/parrots_model_data/objectflow_object/detection_dataset_zqm/rotated_object/',
            # 'sh41:s3://zhangao/rotated_object/',
            # 'sh41:s3://zhangao/rotated_object/',
            # 'sh41:s3://zhangao/rotated_object/',
            # 'sh1984:s3://detection_dataset_zqm/objects365/train/',
            # 'sh1984:s3://detection_dataset_zqm/hoi_dataset/rotated_pro_image/',
            # 'sh1985:s3://zhangao/object_oms_image/',
        ],
        with_ignore_bboxes=True,  # gt_bboxes_ignore
        detect_task=detect_task,  # start from 0
        extra_task=extra_task,
        pipeline=train_pipeline),
    val=test_datasets,
    test=test_datasets,
)
