# model settings
backbone_in_channels =1
backbone_out_channels = 8
backbone_channels = [4,8,8,8]
cls_kernel_channels = 8
cls_search_channels = 8
reg_kernel_channels = 8
reg_search_channels = 8

ratios = [0.5,1.,2.]
scales = [8.]
score_size = 9
stride =4 

template_size = 63
search_size = 95

model = dict(
    type='SiamRPNDW',
    pretrained= None,
    backbone=dict(
        type='AlexNetM',
        in_channels=backbone_in_channels,
        out_channels=backbone_out_channels,
        hidden_channels=backbone_channels,
        max_pool1 = False,
        sync_bn = False,
    ),
    cls_neck=dict(
        type='DepthCorrNeck',
        in_channels=backbone_out_channels,
        kernel_channels=cls_kernel_channels,
        search_channels=cls_search_channels,
        with_bn=False,
        with_relu=False),
    reg_neck=dict(
        type='DepthCorrNeck',
        in_channels=backbone_out_channels,
        kernel_channels=reg_kernel_channels,
        search_channels=reg_search_channels,
        sync_bn=False),
    reg_head=dict(
        type='CBRHead', in_channels=reg_kernel_channels, 
        out_channels=len(ratios) * 4, kernel_size=1,
        sync_bn=False,),
    cls_head=dict(
        type='CBRHead', in_channels=cls_kernel_channels,
        out_channels=len(ratios) * 2, kernel_size=1,
        sync_bn=False,),
    anchor=dict(
        ratios=ratios, scales=scales, score_size=score_size, stride=stride),
    loss_reg=dict(type='SmoothL1Loss', sigma=3.0, weight=75.0),
    loss_cls=dict(
        type='CrossEntropyLoss',
        positive_weight=1.0,
        negative_weight=3.0,
        weight=1.0),
    loss_jit=dict(type='SmoothL1Loss', sigma=3.0, weight=1024),
)
# dataset settings
dataset_type = 'VideoDataset'
template_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        file_client_args=dict(backend='ceph')),
    dict(type='CropTemplate', template_size=template_size, gray=True),
    dict(type='CutOut', prob=0.5),
    dict(type='ImageToTensor', keys=['img']),
]
search_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        file_client_args=dict(backend='ceph')),
    dict(
        type='CropSearchRegion',
        template_size=template_size,
        search_size=search_size,
        crop_ratio=1.5,
        gray=True),
    dict(type='Zoom', max_ratio=0.1),
    dict(type='Shift', max_shift=16),
    dict(type='GaussianBlur', sigma=0.0),
    dict(type='CutOut', prob=0.5),
    dict(type='Jitter', max_jitter=2),
    dict(type='CenterCrop', size=search_size),
    dict(type='CheckBox', thres=1, min_size=2),
    dict(type='ImageToTensor', keys=['img','jitter_img']),
    dict(type='ToTensor', keys=['bbox','jitter_bbox']),
]

anno_root = '/mnt/lustre/share_data/hanyachao/mmtrack/jsons/'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        anno_infos=[
            dict(
                name='300VW',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/300VW_ori/',
                anno=anno_root +
                '300vw.json',
                frame_range=100,
                repeat=100,
                in_class_negative=False,
            ),
            dict(
                name='part1',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/IPC/20191030_监控场景人脸追踪视频采集_part1_chenyanjie/',
                anno=anno_root +
                'part1.json',
                frame_range=4,
                repeat=2,
                in_class_negative=False,
            ),
            dict(
                name='part2',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/IPC/20191031_监控场景人脸追踪视频采集_part2_chenyanjie/',
                anno=anno_root +
                'part2.json',
                frame_range=4,
                repeat=0.8,
                in_class_negative=False,
            ),
            dict(
                name='facial_landmark_image',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/facial_landmark_image/',
                anno=anno_root +
                'facial_landmark_image.json',
                frame_range=1,
                repeat=0.1,
                in_class_negative=False,
            ),
            dict(
                name='LandmarkGhostface',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/facial_landmark_video/',
                anno=anno_root +
                'landmark_ghostface.json',
                frame_range=60,
                repeat=40,
                in_class_negative=False,
            ),
            dict(
                name='LandmarkMidrange',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/facial_landmark_video/',
                anno=anno_root +
                'landmark_midrange.json',
                frame_range=8,
                repeat=2,
                in_class_negative=False,
            ),
            dict(
                name='movie',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/movie/',
                anno=anno_root +
                'movie.json',
                frame_range=1,
                repeat=0.04,
                in_class_negative=False,
            ),
            dict(
                name='BilibiliDance1',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/Bilibili_dance_1/',
                anno=anno_root +
                'bilibili_dance_1.json',
                frame_range=100,
                repeat=40,
                in_class_negative=False,
            ),
            dict(
                name='BilibiliDance2',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/Bilibili_dance_2/',
                anno=anno_root +
                'bilibili_dance_2.json',
                frame_range=20,
                repeat=20,
                in_class_negative=False,
            ),
            dict(
                name='BilibiliOther1',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/Bilibili_other_1/',
                anno=anno_root +
                'bilibili_other_1.json',
                frame_range=20,
                repeat=80,
                in_class_negative=False,
            ),
            dict(
                name='HandNegativeWithFace',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/hand_negative/',
                anno=anno_root +
                'hand_rect_negative_with_Face_sample.json',
                frame_range=1,
                repeat=0.5,
                in_class_negative=False,
            ),
            dict(
                name='HandNegativeNoFace',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/hand_negative/',
                anno=anno_root +
                'hand_rect_negative_no_Face_sample.json',
                frame_range=1,
                repeat=0.5,
                in_class_negative=False,
            ),
            dict(
                name='large_yaw',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/large_angle_data/large_yaw/',
                anno=anno_root +
                'facial_landmark_video_large_yaw_latest_0604_train.json',
                frame_range=16,
                repeat=288,
                in_class_negative=False,
            ),
            dict(
                name='mask_glass',
                root='s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/occlusion_data/20200508_mask_glass_lisiying/',
                anno=anno_root +
                'facial_landmark_video_mask_occlusion_0615.json',
                frame_range=8,
                repeat=8,
                in_class_negative=False,
            ),
        ],
        negative_ratio=0.2,
        template_pipeline=template_pipeline,
        search_pipeline=search_pipeline,
        db_info="/mnt/lustre/share_data/hanyachao/mmtrack/mmtrack",
        ))

# TODO: warm up & group params
total_epochs = 50
warmup_epochs = 10
warmup_base_lr = 1e-4
warmup_end_lr=base_lr = 1e-3
end_lr = 1e-5
# optimizer
optimizer = dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    base_lr=base_lr,
    gamma=pow(end_lr/base_lr, 1/ (total_epochs-warmup_epochs)),
    warmup='exp',
    warmup_iters=warmup_epochs,
    warmup_by_epoch=True,
    warmup_ratio=warmup_base_lr/warmup_end_lr,
    step=list(range(warmup_epochs, total_epochs)))
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='PaviLoggerHook', add_graph=False, add_last_ckpt=True),
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl',
        port=25902)
log_level = 'INFO'
work_dir = './work_dirs/SiamRPNDW'
load_from = None
resume_from = None
workflow = [('train', 1)]
# settings for train/test
train_cfg = None
test_cfg = None
seed = 2018
