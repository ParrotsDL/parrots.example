_base_ = [
    '../configs/face_alignment/facial_landmark_all/models/roi_1.5ms_baseline.py',
    '../configs/face_alignment/facial_landmark_all/datasets/NewMeanPose_v3.py',
    '../configs/face_alignment/facial_landmark_all/schedules/schedule_v3.py',
]
# need to check outside var
crop_size = 192
landmark_num = 282
final_image_format_type = 'GRAY'

# dataset only var
dataset_type="FacialLandmarkFineDataset"
# dataloader_type="AvoidDeadLockDataloader"
dataloader_type="DataLoader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1, degree=10, translate=0.078,
         scale_ratio=0.07),
    dict(type="CloseInnerlip", threshhold=6),
    dict(type="WarpAffineLabel"),
    dict(type="HorizontalFlipLabel", label_type="landmark"),
    dict(type="GetROIRectLabel", needed_label=['mouth', 'eye', 'eyebrow']),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    # dict(type="MotionBlur", Ls=[10, 20], probs=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImagePerturb", shift_noise=1.5),
    dict(type="ImageToTensor", keys=['origin_image', 'perturb_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights', 'gt_rect']),
    dict(type="UtilToTensor", keys=['shift_noise']),
    dict(type="GaussianBlur", mean=0.0, std=0.05),
    dict(type="Collect", image_keys=['origin_image', 'perturb_image'],
         label_keys=['gt_landmarks', 'weights', 'gt_rect'], util_keys=['shift_noise'])
]
validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1),
    dict(type="WarpAffineLabel"),
    dict(type='GetROIRectLabel', needed_label=['mouth', 'eye', 'eyebrow']),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights', 'gt_rect']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_landmarks', 'weights', 'gt_rect'])
]
# test_pipeline = [
#     dict(type="RandomAffineGetMat",input_size=crop_size),
#     dict(type="ToTensor"),
#     dict(type="WarpAffineImage",backend="gpu"),
#     dict(type="Normalize", **img_norm_cfg,backend="gpu"),
#     dict(type="Collect", keys=['img'])
# ]

# lmdb config
train_db_info=dict(
    lmdb_path="/mnt/lustre/share_data/parrots_model_data/sketchv1.4/lmdb/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info=dict(
    lmdb_path="/mnt/lustre/share_data/parrots_model_data/sketchv1.4/lmdb/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

data = dict(
    train=dict(
        type=dataset_type,
        data_infos=[
            dict(
                dataset_name="20181203_close_range_liutinghao",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/Train/20181203_close_range_liutinghao/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
        ],
        data_versions=dict(
            eyelid_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            eyebrow_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            nose_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            lip_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0_forceclose","v2_282pt_AccurateV3_3.0.0"],
            contour_version=["v1_mark", "v1_282pt_AccurateV3_3.0.0", "v1_modelX"],
            iris_version=["v2_mark","v2_282pt_AccurateV3_3.0.0"]
        ),
               
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_infos=[
            dict(
                dataset_name="common_set",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/NewMeanPose/Validate/common_set/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/NewMeanPose/Validate/common_set/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/NewMeanPose/Validate/common_set/Label/",
                source="ceph",
                ceph_clustre="SH1984")
        ],
	    data_versions=dict(
            eyelid_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            eyebrow_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            nose_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            lip_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            contour_version=["v1_mark", "v1_282pt_AccurateV3_3.0.0", "v1_modelX","v1_modelX-v1.0"],
            iris_version=["v2_mark","v2_282pt_AccurateV3_3.0.0"]
        ),
        pipeline=validate_pipeline
    )
)

lb_backbone_channels = [1, 8, 16, 24, 32, 32, 48, 64, 80]
mouth_landmark_num_main = 20
mouth_landmark_num = 64+20
model = dict(
    freeze_main = True,
    cls_path = 'configs/sketch1.4/roi-fix-main/mainbranch_best.pth',
    occ_path = 'configs/sketch1.4/roi-fix-main/mainbranch_best.pth',
    # lip branch
    lip_branch=dict(
        type='Align282RoiBranch_v5',
        roi_start='image',
        feature_map_size=3,
        roi_aug=True,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio=0.05,
        translate_ratio=0.05,
        roialign=dict(type='RoiAlignLayer', pooled_h = 56, pooled_w=56,
                      sampling_ratio=1, spatial_scale=1),
        backbone=dict(
            type='StraightBackbone',
            conv_layer_number=7,
            layers=[dict(type='Conv2dBlock', inchannel=lb_backbone_channels[0],
                         outchannel=lb_backbone_channels[1], kernel_size=5, padding=0, stride=2), # 26
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[1],
                         outchannel=lb_backbone_channels[2], kernel_size=3, padding=0, stride=1), # 24
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[2],
                         outchannel=lb_backbone_channels[3], kernel_size=3, padding=1, stride=2), # 12
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[3],
                         outchannel=lb_backbone_channels[4], kernel_size=3, padding=0, stride=1), # 10
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[4],
                         outchannel=lb_backbone_channels[5], kernel_size=3, padding=0, stride=1), # 8
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[5],
                         outchannel=lb_backbone_channels[6], kernel_size=3, padding=0, stride=1), # 6
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[6],
                         outchannel=lb_backbone_channels[7], kernel_size=3, padding=1, stride=2),], # 3
        ),
        regression_head=dict(
            type='FCHead',
            face_landmark_num = mouth_landmark_num_main,
            conv_layer_number = 1,
            layers=[dict(type='Conv2dBlock',  inchannel=lb_backbone_channels[7],
                        outchannel=lb_backbone_channels[8], kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[128, 128, mouth_landmark_num*2],  # 第二次融合将在此处进行
        ),
    ),
)

# loss
losses = dict(
    losses_info = [
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'mouth',
             part = ['lip_64pt'],
             weight = 128),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'mouth',
             part = ['lip_20pt'],
             weight = 1),
        dict(type = 'DiffLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'mouth',
             part = ['lip_64pt'],
             weight = 6),
        dict(type = 'SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'mouth',
             part = ['lip_64pt'],
             special_feature="Static3D",
             weight = 30),
        dict(type = 'SpecialPointLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'mouth',
             part = ['lip_64pt'],
             special_feature="Inner",
             weight = 3), 
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "mouth",
             part = ['lip_20pt', 'lip_64pt'],
             weight = 1.2,
             loss_name="all_mouth_shift"), # until here, loss of mouth branch
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch = 'eye',
             part = ['eyelid_24pt'],
             weight = 96),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch = 'eye',
             part = ['iris_19pt'],
             weight = 38*1.5),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'eye',
             part = ['eyelid_10pt'],
             weight = 1), 
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "eye",
             part = ['eyelid_24pt', 'iris_19pt', 'eyelid_10pt'],
             weight = 1.2,
             loss_name="all_eye_shift"), # until here, loss of eye branch
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch = 'eyebrow',
             part = ['eyebrow_13pt'],
             weight = 26),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'eyebrow',
             part = ['eyebrow_9pt'],
             weight = 1),
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "eyebrow",
             part = ['eyebrow_13pt','eyebrow_9pt'],
             weight = 1.2,
             loss_name="all_eyebrow_shift"), # until here, loss of left eyebrow branch
    ]
)

metrics = dict(
    metrics_info = [
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["main"],
            part=["face_106pt_all"],
            metric_name="face_106"
        ),
        dict(type="ContourFittingMetric",
            base_metric=dict(type="ContourNMEMetric")),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["mouth"],
            part=["lip_64pt"],
            metric_name="lip_64"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["eye"],
            part=["eyelid_24pt"],
            metric_name="eyelid_24"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["eyebrow"],
            part=["eyebrow_13pt"],
            metric_name="eyebrow_13"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["eye"],
            part=["iris_19pt"],
            metric_name="iris_19"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["mouth","eye","eye","eyebrow"],
            part=["lip_64pt","eyelid_24pt","iris_19pt","eyebrow_13pt"],
            metric_name="roi_part_withiris"
        ),
    ]
)


#data
use_gpu_num = 4
samples_per_gpu=256//use_gpu_num
workers_per_gpu=8
dataloader_type="DataLoader"
dist_params = dict(backend='nccl',port = 21956)
optimizer = dict(type='SGD', lr=0.00015, momentum=0.9, weight_decay=0.0005)
total_epochs = 960//use_gpu_num
total_epochs_percent = total_epochs//10
lr_config = dict(
    gamma=0.5,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    warmup_by_epoch=False,
    step=[total_epochs_percent*5, total_epochs_percent*6, total_epochs_percent*7, total_epochs_percent*8, total_epochs_percent*9])
