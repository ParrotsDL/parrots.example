# outside var
# nme_type = 'STNormDis'

# need to check outside var
crop_size = 112
landmark_num = 106
final_image_format_type = 'GRAY'

# dataset only var
dataset_type="FacialLandmarkConcatDataset"
dataloader_type="Dataloader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.2, degree=20, translate=0.1563,
         scale_ratio=0.06857, flip_ratio=0.5),
    dict(type="WarpAffineLabel"),
    dict(type="HorizontalFlipLabel", label_type="landmark"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    # dict(type="MotionBlur", Ls=[10, 20], probs=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImagePerturb", shift_noise=1.5),
    dict(type="ImageToTensor", keys=['origin_image', 'perturb_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights']),
    dict(type="UtilToTensor", keys=['shift_noise']),
    dict(type="GaussianBlur", mean=0.0, std=0.05),
    dict(type="Collect", image_keys=['origin_image', 'perturb_image'],
         label_keys=['gt_landmarks', 'weights'], util_keys=['shift_noise'])
]
validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.2),
    dict(type="WarpAffineLabel"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_landmarks', 'weights']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_landmarks', 'weights'])
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
    #lmdb_path="s3://parrots_model_data/sketch/liutinghao_1/lmdb_path_1984/sketch/mixlmdb_106_v1_OldMeanPose_0821/",
    lmdb_path="sketch_lmdb",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info=dict(
    #lmdb_path="s3://parrots_model_data/sketch/liutinghao_1/lmdb_path_1984/sketch/mixlmdb_106_v1_OldMeanPose_0821/",
    lmdb_path="sketch_lmdb",
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
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/meitu_2016/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/meitu_2016/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/meitu_2016/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="300W_2016",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/300W_2016/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/300W_2016/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/300W_2016/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="video_frame_2016",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/video_frame_2016/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/video_frame_2016/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/video_frame_2016/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2016",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/ghostface_2016/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/ghostface_2016/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/ghostface_2016/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="sideface_menpo",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/sideface_menpo/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/sideface_menpo/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/sideface_menpo/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="Data_New_106pt",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/Data_New_106pt/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/Data_New_106pt/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/Data_New_106pt/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/ghostface_2/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/ghostface_2/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/ghostface_2/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="squeeze_eye",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/squeeze_eye/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/squeeze_eye/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/squeeze_eye/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="meitu_2016_hand",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/meitu_2016_hand/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/meitu_2016_hand/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/meitu_2016_hand/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="video_frame_2016_hand",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_hand/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_hand/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_hand/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="300W_2016_hand",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/300W_2016_hand/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/300W_2016_hand/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/300W_2016_hand/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2016_hand",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_hand/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_hand/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_hand/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2017_hand",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_hand/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_hand/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_hand/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="sideface_menpo_hand",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_hand/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_hand/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_hand/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="squeeze_eye_hand",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_hand/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_hand/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_hand/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="meitu_2016_mic",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/meitu_2016_mic/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/meitu_2016_mic/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/meitu_2016_mic/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="video_frame_2016_mic",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_mic/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_mic/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_mic/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="300W_2016_mic",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/300W_2016_mic/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/300W_2016_mic/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/300W_2016_mic/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2016_mic",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_mic/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_mic/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_mic/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2017_mic",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_mic/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_mic/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_mic/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="sideface_menpo_mic",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_mic/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_mic/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_mic/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="squeeze_eye_mic",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_mic/json_filelist_0.15.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_mic/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_mic/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="meitu_2016_halfface",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/meitu_2016_halfface/json_filelist_0.1.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/meitu_2016_halfface/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/meitu_2016_halfface/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="video_frame_2016_halfface",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_halfface/json_filelist_0.1.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_halfface/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/video_frame_2016_halfface/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="300W_2016_halfface",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/300W_2016_halfface/json_filelist_0.1.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/300W_2016_halfface/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/300W_2016_halfface/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2016_halfface",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_halfface/json_filelist_0.1.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_halfface/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2016_halfface/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="ghostface_2017_halfface",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_halfface/json_filelist_0.1.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_halfface/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/ghostface_2017_halfface/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="sideface_menpo_halfface",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_halfface/json_filelist_0.1.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_halfface/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/sideface_menpo_halfface/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="squeeze_eye_halfface",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_halfface/json_filelist_0.1.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_halfface/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/OccMake/squeeze_eye_halfface/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="20190929_wink_data",
                repeats=3,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190929_wink_data/json_filelist_select.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/20190929_wink_data/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190929_wink_data/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="20190717_lip_with_mustache",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190717_lip_with_mustache/json_filelist_select.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/20190717_lip_with_mustache/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190717_lip_with_mustache/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="20190808_lip_with_mustache",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190808_lip_with_mustache/json_filelist_select.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/20190808_lip_with_mustache/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190808_lip_with_mustache/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="20190905_duckface_lilei",
                repeats=2,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190905_duckface_lilei/json_filelist_select.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/20190905_duckface_lilei/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190905_duckface_lilei/Label/",
                source="ceph",
                ceph_clustre="SH1984"),
            dict(
                dataset_name="20190318_grin_liukeyi",
                repeats=2,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190318_grin_liukeyi/json_filelist_select.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Train/20190318_grin_liukeyi/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Train/20190318_grin_liukeyi/Label/",
                source="ceph",
                ceph_clustre="SH1984")
        ],
        data_versions=dict(
            eyelid_version=["v1_mark","v1_modelX-v1.0"],
            eyebrow_version=["v1_mark","v1_modelX-v1.0"],
            nose_version=["v1_mark","v1_modelX-v1.0"],
            lip_version=["v1_mark","v1_modelX-v1.0"],
            contour_version=["v1_mark","v1_modelX-v1.0"]
        ),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_infos=[
            dict(
                dataset_name="meitu_2016",
                repeats=1,
                json_file_list="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Validate/meitu_2016/json_filelist.txt",
                image_rootpath="s3://parrots_model_data/sketch/CropFace/OldMeanPose/Validate/meitu_2016/Image/",
                json_rootpath="s3://parrots_model_data/sketch/data/CropFace/OldMeanPose/Validate/meitu_2016/Label/",
                source="ceph",
                ceph_clustre="SH1984",
            ),
        ],
        data_versions=dict(
            eyelid_version=["v1_mark","v1_modelX-v1.0"],
            eyebrow_version=["v1_mark","v1_modelX-v1.0"],
            nose_version=["v1_mark","v1_modelX-v1.0"],
            lip_version=["v1_mark","v1_modelX-v1.0"],
            contour_version=["v1_mark","v1_modelX-v1.0"]
        ),
        pipeline=validate_pipeline
    )
)

# evaluation = dict(interval=2,metrix="nme",main_dataset_name="meitu_2016")
