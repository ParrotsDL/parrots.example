gpus=[0,1,2,3,4,5,6,7]
rank=0
use_pape = True
log_level = 'INFO'
output_dir = './output'
exp_id = 'hrnet32'
log_dir = 'log'

load_from = None
try_load_from=None
resume_from = None
auto_resume = False
print_freq = 10
convert=None
debug=False
pavi_project='default'



# 数据集使用哪几个channel
# 网络监督哪几个channel （位置）
# 网络取出的channel的位置

channel_cfg = dict(
    num_heatmap = 17,
    num_merge_keypoints = 17,
    num_keypoints = [17],

    sub_data_name = ['douyin'],

    model_supervise_channel = [
        [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16],
    ],

    model_select_channel = [
        0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16
    ]

)

# model settings
model = dict(
    type='TopDown',
    pretrained='/mnt/lustre/share_data/star/pretrained_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='PoseHighResolution',
        extra=dict(
            final_conv_kerne=1,
            pretrained_layers=['*'],
            stem_inplanes=64
        ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='basic',
            num_blocks=[4, 4],
            num_channels=[32, 64],
            # num_channels=[48,96],

            fuse_method='sum'
        ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='basic',
            num_blocks=[4, 4, 4],
            num_channels=[32, 64, 128],
            # num_channels=[48,96,192],
            fuse_method='sum'
        ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='basic',
            num_blocks=[4, 4, 4, 4],
            num_channels=[32, 64, 128, 256],
            # num_channels=[48,96,192,384],
            fuse_method='sum'
        ),
    ),

    keypoint_head=dict(
        type='HighResolutionHead',
        pre_stage_channels=32,
        num_joints=17,
        extra=dict(
            final_conv_kerne=1,
            pretrained_layers=['*'],
        ),
    ),
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip'),
    dict(type='HalfBodyTransform'),
    dict(type='RandomScaleRotation'),
    dict(type='AffineTransform'),
    dict(type='GenerateTarget'),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AffineTransform'),
    dict(type='GenerateTarget'),
]

data = dict(
        type = ['CocoDataset'],
        data_cfg = dict(
            train_annotations = [
                '/mnt/lustre/share/DSK/datasets/mscoco2017/annotations/person_keypoints_train2017.json',
                ],

            train_image_path = [
                '/mnt/lustre/share/DSK/datasets/mscoco2017/train2017',
                ],

            valid_annotations = '/mnt/lustre/share/DSK/datasets/mscoco2017/annotations/person_keypoints_val2017.json',
            valid_image_path = '/mnt/lustre/share/DSK/datasets/mscoco2017/val2017',


            world_size = 1,
            use_ceph=False,
            data_format='jpg',
            flip=True,
            rot_factor=40,
            scale_factor=0.5,
            num_joints_half_body=8,
            prob_half_body=0.3,

            color_rgb=True,
            image_size=[192, 256],
            heatmap_size=[48, 64],
            sigma=2,
            target_type='gaussian',
            select_data=False,

            output_dir = output_dir,
            train_pipeline = train_pipeline,
            valid_pipeline = valid_pipeline,
            test_pipeline = valid_pipeline,

            num_merge_keypoints = channel_cfg['num_merge_keypoints'],
            num_keypoints = channel_cfg['num_keypoints'],
            num_heatmap = channel_cfg['num_heatmap'],
            sub_data_name = channel_cfg['sub_data_name'],
            model_supervise_channel = channel_cfg['model_supervise_channel'],
            model_select_channel = channel_cfg['model_select_channel'],

            ))




loss = dict(
    type='JointsMSELoss',
    use_different_joints_weight=False,
    use_target_weight = True,
)

# model training and testing settings
train_cfg = dict(
    trainer=True,
    type='TrainTopDown',

    batch_size_per_gpu=12,
    workers_per_gpu=2,
    shuffle=True,

    begin_epoch=0,
    end_epoch=210,

    optimizer='adam',
    lr=1e-3,
    lr_factor=0.1,
    lr_step=[170, 200],
    wd=0.0001,
    gamma1=0.99,
    momentum=0.9,
    weight_decay=0.0001

)

test_cfg = dict(
    tester=True,
    type='TestTopDown',

    batch_size_per_gpu=16,
    workers_per_gpu=2,
    coco_det_file='',
    coco_bbox_flip='pretrained_models/det/COCO_val2017_detections_AP_H_56_person.json',
    bbox_thre=1.0,
    image_thre=0.0,
    nms_thre=1.0,
    oks_thre=0.9,
    in_vis_thre=0.2,

    soft_nms=False,
    flip_test=True,
    post_process=True,
    shift_heatmap=True,
    use_gt_bbox=True,
)

debug_config = dict(
    debug=True,
    save_images_gt=True,
    save_images_pred=True,
    save_heatmap_gt=True,
    save_heatmap_pred=True
    )
