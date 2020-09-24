# outside var
# nme_type = 'STNormDis'

# need to check outside var
crop_size = 112
landmark_num = 106
final_image_format_type = 'GRAY'

# dataset only var
dataset_type="FacialLandmarkOccDataset"
dataloader_type="Dataloader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.2, degree=20, translate=0.156,
         scale_ratio=0.07, flip_ratio=0.5),
    dict(type="HorizontalFlipLabel", label_type="occlusion"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_occlusion']),
    dict(type="Collect", image_keys=['origin_image'],
         label_keys=['gt_occlusion'])
]
validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.2),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_occlusion']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_occlusion'])
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
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/mixlmdb_106_occ_v1.2/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info=dict(
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/mixlmdb_106_occ_v1.2.4/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

data = dict(
    train=dict(
        type=dataset_type,
        data_infos=[
            # dict(
            #     dataset_name="meitu_2016",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/meitu_2016/json_filelist.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/meitu_2016/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/meitu_2016/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            dict(
                dataset_name="meitu_2016",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Validate/meitu_2016/json_filelist.txt",
                image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Validate/meitu_2016/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Validate/meitu_2016/Label/",
                source="ceph",
                ceph_clustre="sh40_hdd"),
            dict(
                dataset_name="300W_2016",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/300W_2016/json_filelist.txt",
                image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/300W_2016/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/300W_2016/Label/",
                source="ceph",
                ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="mask_occlusion_trans",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/OccMake/mask_occlusion_trans/json_filelist.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/OccMake/mask_occlusion_trans/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/OccMake/mask_occlusion_trans/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="glass_occlusion_trans",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/OccMake/glass_occlusion_trans/json_filelist.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/OccMake/glass_occlusion_trans/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/OccMake/glass_occlusion_trans/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="202003_mask_lilei_7w_all",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/json_filelist_typeC_dark_0.1.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="202003_mask_lilei_7w_all",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/json_filelist_typeC_normal.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="202003_mask_lilei_7w_all",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/json_filelist_typeC_strong_0.25.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="202003_mask_lilei_7w_all",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/json_filelist_typeD_dark_0.1.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="202003_mask_lilei_7w_all",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/json_filelist_typeD_normal.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            # dict(
            #     dataset_name="202003_mask_lilei_7w_all",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/json_filelist_typeD_strong_0.25.txt",
            #     image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Image/",
            #     json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/202003_mask_lilei_7w_all/Label/",
            #     source="ceph",
            #     ceph_clustre="sh40_hdd"),
            dict(
                dataset_name="20200508_mask_glass_lisiying",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/20200508_mask_glass_lisiying/json_filelist.txt",
                image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Train/20200508_mask_glass_lisiying/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/OldMeanPose/Train/20200508_mask_glass_lisiying/Label/",
                source="ceph",
                ceph_clustre="sh40_hdd")

        ],
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_infos=[
            dict(
                dataset_name="meitu_2016",
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/alignment_dataset/Validate/CropFace/OldMeanPose/meitu_2016/json_filelist.txt",
                image_rootpath="s3://ARFace.facial_landmark_bucket/CropFace/OldMeanPose/Validate/meitu_2016/Image/",
                json_rootpath="/mnt/lustre/liutinghao/train_data/alignment_dataset/Validate/CropFace/OldMeanPose/meitu_2016/Label/",
                source="ceph",
                ceph_clustre="sh40_hdd",
            ),
        ],
	    data_versions=dict(
            eyelid_version=["v1_mark","v1_AccruateV3_1.0.1"],
            eyebrow_version=["v1_mark","v1_AccruateV3_1.0.1"],
            nose_version=["v1_mark","v1_AccruateV3_1.0.1"],
            mouth_version=["v1_mark","v1_AccruateV3_1.0.1"],
            contour_version=["v1_mark","v1_AccruateV3_1.0.1"]
        ),
        pipeline=validate_pipeline
    )
)

# evaluation = dict(interval=2,metrix="nme",main_dataset_name="meitu_2016")
