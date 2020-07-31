gpus=[0,1,2,3,4,5,6,7]
rank=0
use_pape = False
log_level = 'INFO'
output_dir = './output'
exp_id = 'acdsph_5017'
log_dir = 'log'

load_from = None
try_load_from=None
resume_from = None
auto_resume = False
print_freq = 10
convert=None
debug=True
pavi_project='default'


channel_cfg = dict(
    num_heatmap = 17,
    num_merge_keypoints = 17,
    num_keypoints = [17],

    sub_data_name = ['coco'],

    model_supervise_channel = [
        [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16],
    ],

    model_select_channel = [
        0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16
        # 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    ]

)

model = dict(
    type='BottomUp',
    pretrained='',
    backbone=dict(
        type='sunyan_light_model_bn',
        #type='JnetOri_BU',
        #block = 'Bottleneck',
        #layers = [2, 3, 4, 1],
        out_channels = channel_cfg['num_heatmap']*2,
    ),
    keypoint_head=dict(
        type='NoneConv',
    #    pre_stage_channels=256,
        num_joints=17,
        tag_per_joint=True,
    #    extra=dict(
    #        final_conv_kerne=1,
    #        pretrained_layers=['*'],
    #    ),
    #    loss=dict(
    #        with_ae_loss=True, #[True]
    #    )
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomAffineTransform'),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize'),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='ToTensor'),
    # dict(type='Normalize'),
]

data = dict(
        type = ['BU_CocoDataset'], #, 'CocoDataset'],
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
            num_scales=1,

            data_format='jpg',
            flip=0.5,
            max_num_people=30,
            rot_factor=30,
            scale_type='short',
            scale_factor=[0.75,1.5],
            max_translate=40,
            scale_aware_sigma=False,
            color_rgb=True,
            image_size=[512,512],
            heatmap_size=[64],#[128,256]
            sigma=2,
            output_dir = output_dir,
            train_pipeline = train_pipeline,
            valid_pipeline = valid_pipeline,
            test_pipeline = valid_pipeline,


            num_merge_keypoints = channel_cfg['num_merge_keypoints'],
            num_keypoints = channel_cfg['num_keypoints'],
            num_joints = channel_cfg['num_merge_keypoints'],
            num_heatmap = channel_cfg['num_heatmap'],
            sub_data_name = channel_cfg['sub_data_name'],
            model_supervise_channel = channel_cfg['model_supervise_channel'],
            model_select_channel = channel_cfg['model_select_channel'],
            ))

loss = dict(
    type='MultiLossFactory',
    num_stages=1,#2
    ae_loss_type='exp',
    with_ae_loss=[True],
    push_loss_factor=[0.001],
    pull_loss_factor=[0.001],
    with_heatmaps_loss=[True],#[True,True]
    heatmaps_loss_factor=[1.0],

)


# model training and testing settings
train_cfg = dict(
    trainer=True,
    type='TrainBottomUp',
    batch_size_per_gpu=16,
    workers_per_gpu=1,
    shuffle=True,

    begin_epoch=0,
    end_epoch=300,

    optimizer='adam',
    lr=0.00015,
    lr_factor=0.1,
    lr_step=[200,260],
    # lr_step=[90,120],
    wd=0.0001,
    gamma1=0.99,
    momentum=0.9,
    weight_decay=0.0001

)
test_cfg = dict(
    tester=False,
    type='TestBottomUp',
    batch_size_per_gpu=1,
    workers_per_gpu=1,
    coco_det_file='',
    coco_bbox_flip='pretrained_models/det/COCO_val2017_detections_AP_H_56_person.json',
    bbox_thre=1.0,
    image_thre=0.0,
    nms_thre=1.0,
    oks_thre=0.9,
    in_vis_thre=0.2,

    detection_threshold=0.1,
    scale_factor=[1],
    with_heatmaps=[True],
    with_ae=[True],
    project2image=True,
    nms_kernel=5,
    nms_padding=2,
    tag_threshold=1,
    use_detection_val=True,
    ignore_too_much=False,
    adjust=True,
    refine=True,

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
