# outside var
# nme_type = 'STNormDis'

# need to check outside var
crop_size = 112
landmark_num = 106
final_image_format_type = 'GRAY'

# dataset only var
dataset_type="FacialLandmarkFineDataset"
dataloader_type="AvoidDeadLockDataLoader"
# dataloader_type="DataLoader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
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
]
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
# test_pipeline = [
#     dict(type="RandomAffineGetMat",input_size=crop_size),
#     dict(type="ToTensor"),
#     dict(type="WarpAffineImage",backend="gpu"),
#     dict(type="Normalize", **img_norm_cfg,backend="gpu"),
#     dict(type="Collect", keys=['img'])
# ]

# lmdb config
train_db_info=dict(
    lmdb_path="/mnt/lustre/chenzukai/base_training_code/lmdb_path/sketch/v1.4_lmdb_ssd_split/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info=dict(
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/v1.4_lmdb_ssd_split/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)


data = dict(
    train=dict(
        type=dataset_type,
        data_infos=[
            dict(
            	dataset_name="20181203_close_range_liutinghao_pair_debug10",
            	repeats=1,
                json_file_list="/mnt/lustre/chenzukai/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/json_filelist_pair.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Image/",
            	json_rootpath="/mnt/lustreold/lisiying1/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20180821_middle_range_15_sunkeqiang",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20180821_middle_range_15_sunkeqiang/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20180821_middle_range_15_sunkeqiang/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20180821_middle_range_15_sunkeqiang/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20180823_middle_range_12_sunkeqiang",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20180823_middle_range_12_sunkeqiang/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20180823_middle_range_12_sunkeqiang/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20180823_middle_range_12_sunkeqiang/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20180910_omron_eyelid_data_liutinghao",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20180910_omron_eyelid_data_liutinghao/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20180910_omron_eyelid_data_liutinghao/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20180910_omron_eyelid_data_liutinghao/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20181127_ghostface_liutinghao",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181127_ghostface_liutinghao/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20181127_ghostface_liutinghao/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181127_ghostface_liutinghao/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20181213_middle_range_yaw_pitch_liutinghao",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181213_middle_range_yaw_pitch_liutinghao/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20181213_middle_range_yaw_pitch_liutinghao/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181213_middle_range_yaw_pitch_liutinghao/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20190202_multi_degree_joker_face_liutinghao",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190202_multi_degree_joker_face_liutinghao/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20190202_multi_degree_joker_face_liutinghao/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190202_multi_degree_joker_face_liutinghao/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20190929_wink_data",
            	repeats=3,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190929_wink_data/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20190929_wink_data/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190929_wink_data/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20190717_lip_with_mustache",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190717_lip_with_mustache/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20190717_lip_with_mustache/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190717_lip_with_mustache/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20190808_lip_with_mustache",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190808_lip_with_mustache/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20190808_lip_with_mustache/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190808_lip_with_mustache/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20190905_duckface_lilei",
            	repeats=3,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190905_duckface_lilei/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20190905_duckface_lilei/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190905_duckface_lilei/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20190318_grin_liukeyi",
            	repeats=2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190318_grin_liukeyi/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20190318_grin_liukeyi/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20190318_grin_liukeyi/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="close_range_halfface",
            	repeats=0.1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/close_range_halfface/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/close_range_halfface/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/close_range_halfface/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_range_15_halfface",
            	repeats=0.1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_15_halfface/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_range_15_halfface/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_15_halfface/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_range_12_halfface",
            	repeats=0.1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_12_halfface/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_range_12_halfface/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_12_halfface/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="omron_halfface",
            	repeats=0.1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/omron_halfface/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/omron_halfface/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/omron_halfface/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="ghostface_halfface",
            	repeats=0.1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/ghostface_halfface/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/ghostface_halfface/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/ghostface_halfface/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_yaw_halfface",
            	repeats=0.1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_yaw_halfface/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_yaw_halfface/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_yaw_halfface/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="multi_degree_joker_halfface",
            	repeats=0.1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_halfface/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_halfface/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_halfface/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="close_range_mic",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/close_range_mic/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/close_range_mic/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/close_range_mic/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_range_15_mic",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_15_mic/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_range_15_mic/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_15_mic/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_range_12_mic",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_12_mic/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_range_12_mic/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_12_mic/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="omron_mic",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/omron_mic/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/omron_mic/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/omron_mic/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="ghostface_mic",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/ghostface_mic/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/ghostface_mic/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/ghostface_mic/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_yaw_mic",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_yaw_mic/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_yaw_mic/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_yaw_mic/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="multi_degree_joker_mic",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_mic/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_mic/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_mic/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="close_range_hand",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/close_range_hand/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/close_range_hand/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/close_range_hand/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_range_15_hand",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_15_hand/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_range_15_hand/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_15_hand/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_range_12_hand",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_12_hand/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_range_12_hand/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_range_12_hand/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="omron_hand",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/omron_hand/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/omron_hand/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/omron_hand/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="ghostface_hand",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/ghostface_hand/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/ghostface_hand/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/ghostface_hand/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_yaw_hand",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_yaw_hand/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_yaw_hand/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/middle_yaw_hand/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="multi_degree_joker_hand",
            	repeats=0.2,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_hand/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_hand/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/OccMake/multi_degree_joker_hand/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="20191216_meitu_2016",
            	repeats=1,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20191216_meitu_2016/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20191216_meitu_2016/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20191216_meitu_2016/Label/",
            	source="ceph",
            	ceph_clustre="sh40_ssd"),
	    	dict(
                dataset_name="20201109_closemouth_170open",
                repeats=2,
                json_file_list="/mnt/lustre/lisiying1/data/CropFace/NewMeanPose/Train/20201109_closemouth_170open/json_filelist.txt",
                image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/20201109_closemouth_170open/Image/",
                json_rootpath="/mnt/lustre/lisiying1/data/CropFace/NewMeanPose/Train/20201109_closemouth_170open/Label/",
                source="ceph",
                ceph_clustre="sh40_ssd"),
	    	dict(
                dataset_name="close_range_hair",
                repeats=0.5,
                json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/close_range_hair/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181203_close_range_liutinghao/Label/",
                source="ceph",
                ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="ghostface_hair",
            	repeats=0.5,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181127_ghostface_liutinghao/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/ghostface_hair/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181127_ghostface_liutinghao/Label/",
            	source="ceph",
                ceph_clustre="sh40_ssd"),
            dict(
            	dataset_name="middle_yaw_hair",
            	repeats=0.5,
            	json_file_list="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181213_middle_range_yaw_pitch_liutinghao/json_filelist.txt",
            	image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Train/OccMake/middle_yaw_hair/Image/",
            	json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Train/20181213_middle_range_yaw_pitch_liutinghao/Label/",
            	source="ceph",
                ceph_clustre="sh40_ssd"),	
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
                image_rootpath="s3://ARFace.Facial_Landmark_Bucket/CropFace/NewMeanPose/Validate/common_set/Image/",
                json_rootpath="/mnt/lustre/lisiying1new/data/CropFace/NewMeanPose/Validate/common_set/Label/",
                source="ceph",
                ceph_clustre="sh40_ssd")
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

# evaluation = dict(interval=2,metrix="nme",main_dataset_name="meitu_2016")
