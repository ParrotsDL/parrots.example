epoch_size = 10000
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10*epoch_size, 20*epoch_size])
evaluation = dict(interval=1000, 
                  metric=dict(bbox_head=dict(metric='RatP', target_prec=0.99, iou_thr=0.3)))
max_iters = epoch_size * 14
workflow = [('train', max_iters)]

# task_groups = {'face': [i for i in range(0, 8)],
#                'body': [i for i in range(8, 16)],
#                'facebody': [i for i in range(0, 16)],
#                'child': [i for i in range(16, 24)]}
# task_prefix = {'neck': 'facebody',
#                'bbox_head.cls_convs': 'facebody',
#                'bbox_head.reg_convs': 'facebody',
#                'bbox_head.retina_reg': 'facebody',
#                'bbox_head.retina_cls': 'body'}
task_groups = {'catdog': [i for i in range(0, 8)]}
task_prefix = {'neck': 'catdog',
               'bbox_head.cls_convs': 'catdog',
               'bbox_head.reg_convs': 'catdog',
               'bbox_head.retina_reg': 'catdog',
               'bbox_head.retina_cls': 'catdog'}

checkpoint_config = dict(interval=epoch_size)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/mnt/lustre/datatag/VACabin/pretrain/merged_facebody_standard_det_color_dfpn_128_32_try2_iter_240000_body.pth'
load_from = None
resume_from = None

model = dict(
    type='RetinaNet',
    pretrained='/mnt/lustre/share_data/parrots_model_ckpt/objectflow_pet/mb2_1x_gray_top1_71.18_top5_89.93.pth.tar',
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
data_root_pmdb_train = '/mnt/lustre/share_data/parrots_model_data/objectflow_pet/pet_data/'
data_root_pmdb_test = '/mnt/lustre/share_data/parrots_model_data/objectflow_pet/pet_data/'

detect_task = dict(pet=0)
extra_task = []

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color', file_client_args=dict(backend='petrel')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomWarpAffine', scale=(0.8, 1.2), min_bbox_size=5),
    dict(type='Resize', img_scale=[(1280, 736), (960, 544), (640, 384)], keep_ratio=True),
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
            dict(type='Collect', keys=['img'])
        ])
]
test_datasets = [
    dict(
        type='CustomTaskDataset',
        ann_file=data_root_pmdb_test + 'test_1.pkl',
        img_prefix='s3://parrots_model_data/objectflow_pet/dms_pet/final_test/',
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
        type='CustomTaskDataset',
        ann_file=[
            data_root_pmdb_train + 'coco_train.pkl',
            data_root_pmdb_train + 'open_train.pkl',
            # data_root_pmdb_train + 'bilibili640_train.pkl',
            # data_root_pmdb_train + 'new_cat_crawler_train.pkl',
            # data_root_pmdb_train + 'new_dog_crawler_train.pkl',
            # data_root_pmdb_train + 'pet_dms_1202-1207.pkl',
            # data_root_pmdb_train + 'pet_dms_1208-1215_train.pkl',
        ],
        img_prefix=[
            's3://parrots_model_data/objectflow_pet/cat_dog/cat_dog_train/',
            's3://parrots_model_data/objectflow_pet/cat_dog/cat_dog_open/',
            # 'sh1985:s3://pet_dataset_xme/mnt/lustre/xumengen1/DetectionTools/cat_dog_bilibili/',
            # 'sh1985:s3://pet_dataset_xme/mnt/lustre/datatag/xumengen1/other_data/Image/cats/',
            # 'sh1985:s3://pet_dataset_xme/mnt/lustre/datatag/xumengen1/other_data/Image/dogs/',
            # 'sh1985:s3://pet_dataset_xme/mnt/lustre/datatag/xumengen1/dms_pet/Image_20201202-20201207_train/',
            # 'sh1985:s3://pet_dataset_xme/mnt/lustre/datatag/xumengen1/dms_pet/Image_20201208-20201215_train/',
        ],
        with_ignore_bboxes=True,
        detect_task=detect_task,
        extra_task=extra_task,
        pipeline=train_pipeline),
    val=test_datasets,
    test=test_datasets,
  ) 
