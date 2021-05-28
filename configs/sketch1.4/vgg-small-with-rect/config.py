_base_ = [
    '../configs/face_alignment/facial_landmark_main/models/vgg_small_with_rect.py',
    '../configs/face_alignment/facial_landmark_main/datasets/NewMeanPose_v3_with_rect.py',
    '../configs/face_alignment/facial_landmark_main/schedules/schedule_v3.py',
]

dist_params = dict(backend='nccl', port=20223)
# need to check outside var
crop_size = 192
landmark_num = 106
final_image_format_type = 'GRAY'

# dataset only var
dataset_type="FacialLandmarkFineDataset"
dataloader_type="DataLoader"
# dataloader_type="AvoidDeadLockDataloader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1, degree=10, translate=0.059,
         scale_ratio=0.06857, flip_ratio=0.5),
    dict(type="WarpAffineLabel"),
    dict(type="HorizontalFlipLabel", label_type="landmark"),
    dict(type="GetROIRectLabel",needed_label=['mouth','eye','eyebrow'],mouth_expand_ratio=1.4),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImagePerturb", shift_noise=1.5),
    dict(type="ImageToTensor", keys=['origin_image', 'perturb_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights','gt_rect']),
    dict(type="UtilToTensor", keys=['shift_noise']),
    dict(type="GaussianBlur", mean=0.0, std=0.05),
    dict(type="Collect", image_keys=['origin_image', 'perturb_image'],
         label_keys=['gt_landmarks', 'weights','gt_rect'], util_keys=['shift_noise'])
]
validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1),
    dict(type="WarpAffineLabel"),
    dict(type="GetROIRectLabel",needed_label=['mouth','eye','eyebrow'],mouth_expand_ratio=1.4),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights','gt_rect']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_landmarks', 'weights','gt_rect'])
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
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/Train/20181203_close_range_liutinghao/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Label/",
            	source="ceph",
            	ceph_clustre="SH1984"),
        ],
        data_versions=dict(
        	eyelid_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
        	eyebrow_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
        	nose_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0"],
        	lip_version=["v2_mark", "v2_282pt_AccurateV3_3.0.0","v2_282pt_AccurateV3_3.0.0_forceclose"],
        	contour_version=["v1_mark", "v1_282pt_AccurateV3_3.0.0", "v1_modelX-v1.0", "v1_modelX"],
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
            iris_version=["v2_mark","v2_282pt_AccurateV3_3.0.0"]
        ),
        pipeline=validate_pipeline
    )
)

lr_config = dict(
    gamma=0.5,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[150, 180, 210, 240, 270])
total_epochs = 300

# metrics = dict(
#     metrics_info = [
#         dict(type="LandmarkMetric",
#             base_metric=dict(type="BaseNMEMetric"),
#             branch=["main"],
#             part=["nose_15pt"],
#             metric_name="nose_nme"),
#         dict(type="LandmarkMetric",
#             base_metric=dict(type="BaseNMEMetric"),
#             branch=["main"],
#             part=["lip_20pt"],
#             metric_name="lip_nme"),
#         dict(type="LandmarkMetric",
#             base_metric=dict(type="BaseNMEMetric"),
#             branch=["main"],
#             part=["contour_33pt"],
#             metric_name="contour_nme"),
#         dict(type = "RectMetric",
#              base_metric=dict(type="BaseNMEMetric"),
#              rects = ['mouth','eye','eyebrow'],
#              metric_name="all_rect"),
#         dict(type = "ContourFittingMetric",
#              base_metric=dict(type="ContourNMEMetric")),
#         dict(type="LandmarkMetric",
#             base_metric=dict(type="BaseNMEMetric"),
#             branch=["main"],
#             part=["face_106pt_all"],
#             metric_name="face_106_nme")
#     ]
# )