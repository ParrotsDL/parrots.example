
gpus=[0,1,2,3,4,5,6,7]
rank=0
use_pape = False
log_level = 'INFO'
output_dir = './output'
exp_id = 'acdsph_Jnet'
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
        # 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    ]

)

model = dict(
    type='TopDown',
    pretrained='',
    backbone=dict(
        type='sunyan_light_model_bn',
        out_channels=17,
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
            rot_factor=15,
            scale_factor=0.3,
            num_joints_half_body=8,
            prob_half_body=0.0,

            color_rgb=True,
            image_size=[192,256],
            heatmap_size=[24,32],

            # image_size=[144,112],
            # heatmap_size=[18,14],

            # image_size=[288,384],
            # heatmap_size=[72,96],
            # image_size=[96,96],
            # heatmap_size=[12,12],
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

    batch_size_per_gpu=64,
    # batch_size_per_gpu=512,
    workers_per_gpu=2,
    shuffle=True,

    begin_epoch=0,
    end_epoch=1,

    optimizer='adam',
    lr=2e-4,
    lr_factor=0.1,
    lr_step=[140,180],
    # lr_step=[90,120],
    wd=0.0001,
    gamma1=0.99,
    momentum=0.9,
    weight_decay=0.0001
    
)


test_cfg = dict(
    tester=True,
    type='TestTopDown',

    batch_size_per_gpu=64,
    workers_per_gpu=2,
    coco_det_file='',
    coco_bbox_flip='/mnt/lustre/share_data/star/COCO_val2017_detections_AP_H_56_person.json',
    bbox_thre=1.0,
    image_thre=0.0,
    nms_thre=1.0,
    oks_thre=0.9,
    in_vis_thre=0.2,


    soft_nms=False,
    flip_test=False,
    post_process=True,
    shift_heatmap=True,
    use_gt_bbox=False,
)

debug_config = dict(
    debug=True,
    save_images_gt=True,
    save_images_pred=True,
    save_heatmap_gt=True,
    save_heatmap_pred=True
    )







