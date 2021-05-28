_base_ = [
    '../configs/face_alignment/facial_landmark_main/models/vgg_small.py',
    '../configs/face_alignment/facial_landmark_main/datasets/NewMeanPose_v3_MotionMake_pair_multi_dataloader.py',
    '../configs/face_alignment/facial_landmark_main/schedules/schedule_v3.py',
]

samples_per_gpu = [64, 32]

model = dict(
    type='TinyAlign106_backbone_pair_hanxy',
)

# need to check outside var
crop_size = 112
landmark_num = 106
final_image_format_type = 'GRAY'

# dataset only var
multi_dataloader = True
train_dataset_type = ["FacialLandmarkFineDataset", "FacialLandmarkFinePairDataset"]
val_dataset_type = "FacialLandmarkFineDataset"
dataloader_type="DataLoader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = list()
train_pipeline.append([
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1, degree=10, translate=0.059,
         scale_ratio=0.06857, flip_ratio=0.5),
    dict(type="WarpAffineLabel"),
    dict(type="HorizontalFlipLabel", label_type="landmark"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImagePerturb", shift_noise=1.5),
    dict(type="ImageToTensor", keys=['origin_image', 'perturb_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights']),
    dict(type="UtilToTensor", keys=['shift_noise']),
    dict(type="GaussianBlur", mean=0.0, std=0.05),
    dict(type="Collect", image_keys=['origin_image', 'perturb_image'],
         label_keys=['gt_landmarks', 'weights'], util_keys=['shift_noise'])
])
train_pipeline.append([
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1, degree=10, translate=0.059,
         scale_ratio=0.06857, flip_ratio=0.5),
    dict(type="WarpAffineLabel"),
    dict(type="HorizontalFlipLabelPair", label_type="landmark"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ColorConvertPair", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImagePair"),
    dict(type="NormalizePair", **img_norm_cfg),
    dict(type="ImagePerturb", shift_noise=1.5),
    dict(type="ImageToTensor", keys=['origin_image', 'pair_image', 'perturb_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'move_weights', 'weights']),
    dict(type="UtilToTensor", keys=['shift_noise']),
    dict(type="GaussianBlur", mean=0.0, std=0.05),
    dict(type="Collect", image_keys=['origin_image', 'pair_image', 'perturb_image'],
         label_keys=['gt_landmarks', 'move_weights', 'weights'], util_keys=['shift_noise'])
])

validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1),
    dict(type="WarpAffineLabel"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_landmarks', 'weights'])
]

# lmdb config
train_db_info = list()
train_db_info.append(dict(
    lmdb_path="/mnt/lustre/share_data/parrots_model_data/sketchv1.4/lmdb/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
))
train_db_info.append(dict(
    lmdb_path="/mnt/lustre/share_data/parrots_model_data/sketchv1.4/lmdb/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
))

validate_db_info=dict(
    lmdb_path="/mnt/lustre/share_data/parrots_model_data/sketchv1.4/lmdb/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

data = dict(
    train=[
        dict(
            type=train_dataset_type[0],
            data_infos=[
                dict(
                    dataset_name="20181203_close_range_liutinghao",
                    repeats=1,
                    json_file_list="/mnt/lustreold/lisiying1/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/json_filelist.txt",
                    image_rootpath="s3://parrots_model_data/sketch/Train/20181203_close_range_liutinghao/Image/",
                    json_rootpath="/mnt/lustreold/lisiying1/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Label/",
                    source="ceph",
                    ceph_clustre="SH1984"),
            ],
            data_versions=dict(
                eyelid_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
                eyebrow_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
                nose_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
                lip_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0", "v2_282pt_AccurateV3_3.0.0_forceclose"],
                contour_version=["v1_mark", "v1_282pt_AccurateV3_3.0.0", "v1_modelX-v1.0", "v1_modelX"],
                iris_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"]
            ),
            pipeline=train_pipeline[0]
        ),
        dict(
            type=train_dataset_type[1],
            data_infos=[
                dict(
                    dataset_name="close_range_eyeclose_pair",
                    repeats=1,
                    json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/Train/Replaced/20181203-close_range_high_pixel_face_data-liutinghao/renew_all_oldaffine/json_filelist.txt",
#                    image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Image/",
                    image_rootpath="s3://parrots_model_data/sketch/Train/20181203_close_range_liutinghao/Image/",
                    json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Label/",
                    #image_pair_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/MotionMake/close_range_eyeclose/Image/",
                    image_pair_rootpath="s3://parrots_model_data/sketch/Train/close_range_eyeclose/Image/",
                    json_pair_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/Train/Replaced/20181203-close_range_high_pixel_face_data-liutinghao/renew_all_oldaffine/Label/",
                    source="ceph",
                    ceph_clustre="SH1984"),
            ],
            data_versions=dict(
                eyelid_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
                eyebrow_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
                nose_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
                lip_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0", "v2_282pt_AccurateV3_3.0.0_forceclose"],
                contour_version=["v1_mark", "v1_282pt_AccurateV3_3.0.0", "v1_modelX-v1.0", "v1_modelX"],
                iris_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"]
            ),
            pipeline=train_pipeline[1]
        ),
    ],
    val=dict(
        type=val_dataset_type,
        data_infos=[
            dict(
                dataset_name="common_set",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Validate/common_set/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/NewMeanPose/Validate/common_set/Image/",
                json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Validate/common_set/Label/",
                source="ceph",
                ceph_clustre="SH1984")
        ],
        data_versions=dict(
            eyelid_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            eyebrow_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            nose_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            lip_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
            contour_version=["v1_mark", "v1_282pt_AccurateV3_3.0.0", "v1_modelX-v1.0", "v1_modelX"],
            iris_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"]
        ),
        pipeline=validate_pipeline
    )
)

# loss
losses = dict(
    losses_info=[
        # main loss
        dict(type='LandmarkLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch='main',
             part=['eyebrow_9pt', 'eyelid_10pt', 'nose_15pt', 'lip_20pt'],
             weight=73,
             loss_name="face_organs"),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch='main',
             part=['contour_33pt'],
             special_feature="Curve",
             weight=15,
             loss_name="face_contour"),
        # cf loss
        dict(type="ContourFittingLoss",
             base_loss=dict(type="CF_SoftWingLoss"),
             weight=30),
        # 3d loss
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['nose_15pt'],
             special_feature="Static3D",
             weight=2),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['eyebrow_9pt'],
             special_feature="Static3D",
             weight=12),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['lip_20pt'],
             special_feature="Static3D",
             weight=24),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['contour_33pt'],
             special_feature="Static3D",
             weight=12),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['eyelid_10pt'],
             special_feature="Static3D",
             weight=20),
        # shift loss
        dict(type='ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             weight=1.5 * 0.7),
        dict(type='ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             suffix="pair",
             weight=1.5 * 0.3),
        # pair loss
        dict(type='PairLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             weight=1.5),
        # diff loss
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['left_eyelid_10pt'],
             weight=15,
             loss_name="left_eye_diff"),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['right_eyelid_10pt'],
             weight=15,
             loss_name="right_eye_diff"),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['lip_20pt'],
             diff_part="inner",
             weight=10,
             loss_name="inner_lip_diff"),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['lip_20pt'],
             weight=5,
             loss_name="lip_diff")
    ]
)

dist_params = dict(backend='nccl', port=80215)

lr_config = dict(
    gamma=0.5,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[150, 180, 210, 240, 270])
total_epochs = 300
