# outside var
# nme_type = 'STNormDis'

# need to check outside var
crop_size = 256
landmark_num = 282
final_image_format_type = 'GRAY'

# dataset only var
dataset_type="FacialLandmarkFineDataset"
dataloader_type="Dataloader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.2, degree=20, translate=0.156,
         scale_ratio=0.07),
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
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.2),
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
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/mixlmdb_282_v3.0_test_0821/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info=dict(
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/mixlmdb_282_v3.0_test_0821/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

data = dict(
    train=dict(
        type=dataset_type,
        data_infos=[
            dict(
                dataset_name="meitu_2016",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/20191216_meitu_2016/json_filelist.txt",
                image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/20191216_meitu_2016/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/20191216_meitu_2016/Label/",
                source="ceph",
                ceph_clustre="sh40_hdd"),
            dict(
                dataset_name="middle_range_15_halfface",
                repeats=0.1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/OccMake/middle_range_15_halfface/json_filelist.txt",
                image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/OccMake/middle_range_15_halfface/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/OccMake/middle_range_15_halfface/Label/",
                source="ceph",
                ceph_clustre="sh40_hdd")
        ],
        data_versions=dict(
            eyelid_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            eyebrow_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            nose_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            lip_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            contour_version=["v1_mark", "v1_AccruateV3_3.0.0", "v1_modelX-v1.0"],
            iris_version=["v2_mark","v2_AccruateV3_3.0.0"]
        ),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_infos=[
            dict(
                dataset_name="meitu_2016",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Validate/20191216_meitu_2016/json_filelist.txt",
                image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Validate/20191216_meitu_2016/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Validate/20191216_meitu_2016/Label/",
                source="ceph",
                ceph_clustre="sh40_hdd",
            ),
        ],
	    data_versions=dict(
            eyelid_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            eyebrow_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            nose_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            lip_version=["v2_mark", "v2_AccruateV3_3.0.0"],
            contour_version=["v1_mark", "v1_AccruateV3_3.0.0", "v1_modelX-v1.0"],
            iris_version=["v2_mark","v2_AccruateV3_3.0.0"]
        ),
        pipeline=validate_pipeline
    )
)

# evaluation = dict(interval=2,metrix="nme",main_dataset_name="meitu_2016")
