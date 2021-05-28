# need to check outside var
crop_size = 192
landmark_num = 282
num_output = 2
final_image_format_type = 'GRAY'

# dataset only var
dataset_type = "EyeStateWholeFaceDataset"
dataloader_type = "DataLoader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1, degree=10, translate=0.078,
         scale_ratio=0.07),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    # dict(type="MotionBlur", Ls=[40, 75], probs=0.15),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_eyestate']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_eyestate'])
]

validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.1),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_eyestate']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_eyestate'])
]

# lmdb config
train_db_info = dict(
    lmdb_path="/mnt/lustre/hanxiaoyang/lmdb_path/sketch/iris_train_newmeanpose/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info = dict(
    lmdb_path="/mnt/lustre/hanxiaoyang/lmdb_path/sketch/iris_test/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

data = dict(
    train=dict(
        type=dataset_type,
        image_size=dict(whole_size=384, center_size=256, expand_ratio=0.15),
        label_format="txt",
        data_infos=[
            dict(
                dataset_name="iris_0",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20181203-close_range_high_pixel_face_data-liutinghao_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20181203-close_range_high_pixel_face_data-liutinghao/renew_all_oldaffine/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_1",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20180821_High_resolution_sunkeqiang_15_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20180821_High_resolution_sunkeqiang_15/renew_all_oldaffine/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_2",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20180823_High_resolution_sunkeqiang_12_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20180823_High_resolution_sunkeqiang_12/renew_all_oldaffine/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_3",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20180910_omron_eyelid_data_liutinghao_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20180910_omron_eyelid_data_liutinghao/renew_all_oldaffine/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_4",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20181127_ghostface_high_reso_liutinghao_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20181127_ghostface_high_reso_liutinghao/renew_all_oldaffine/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_5",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20181213-middle_range_yaw_and_pitch_photo_liutinghao_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20181213-middle_range_yaw_and_pitch_photo_liutinghao/renew_all_oldaffine/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_6",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20190929_wink_data_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20190929_wink_data/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_7",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20190717_lip_with_mustache_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20190717_lip_with_mustache/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_8",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20190808_lip_with_mustache_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20190808_lip_with_mustache/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_9",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20190905_duckface_lilei_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20190905_duckface_lilei/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_10",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/20190318_grin_liukeyi_eyestate.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/20190318_grin_liukeyi/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_0",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_right_eye/20181127_ghostface_high_reso_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_right_eye/20181127_ghostface_high_reso_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_1",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_right_eye/20180823_High_resolution_sunkeqiang_12/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_right_eye/20180823_High_resolution_sunkeqiang_12/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_2",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_right_eye/20181203-close_range_high_pixel_face_data-liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_right_eye/20181203-close_range_high_pixel_face_data-liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_3",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_right_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_right_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_4",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_right_eye/20180821_High_resolution_sunkeqiang_15/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_right_eye/20180821_High_resolution_sunkeqiang_15/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_5",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_right_eye/20180910_omron_eyelid_data_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_right_eye/20180910_omron_eyelid_data_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_6",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_right_eye/20181127_ghostface_high_reso_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_right_eye/20181127_ghostface_high_reso_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_7",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_right_eye/20180823_High_resolution_sunkeqiang_12/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_right_eye/20180823_High_resolution_sunkeqiang_12/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_8",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_right_eye/20181203-close_range_high_pixel_face_data-liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_right_eye/20181203-close_range_high_pixel_face_data-liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_9",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_right_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_right_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_10",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_right_eye/20180821_High_resolution_sunkeqiang_15/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_right_eye/20180821_High_resolution_sunkeqiang_15/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_11",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_right_eye/20180910_omron_eyelid_data_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_right_eye/20180910_omron_eyelid_data_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_12",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_left_eye/20181127_ghostface_high_reso_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_left_eye/20181127_ghostface_high_reso_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_13",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_left_eye/20180823_High_resolution_sunkeqiang_12/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_left_eye/20180823_High_resolution_sunkeqiang_12/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_14",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_left_eye/20181203-close_range_high_pixel_face_data-liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_left_eye/20181203-close_range_high_pixel_face_data-liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_15",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_left_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_left_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_16",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_left_eye/20180821_High_resolution_sunkeqiang_15/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_left_eye/20180821_High_resolution_sunkeqiang_15/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_17",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_finger_in_left_eye/20180910_omron_eyelid_data_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_finger_in_left_eye/20180910_omron_eyelid_data_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_18",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_left_eye/20181127_ghostface_high_reso_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_left_eye/20181127_ghostface_high_reso_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_19",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_left_eye/20180823_High_resolution_sunkeqiang_12/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_left_eye/20180823_High_resolution_sunkeqiang_12/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_20",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_left_eye/20181203-close_range_high_pixel_face_data-liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_left_eye/20181203-close_range_high_pixel_face_data-liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_21",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_left_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_left_eye/20181213-middle_range_yaw_and_pitch_photo_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_22",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_left_eye/20180821_High_resolution_sunkeqiang_15/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_left_eye/20180821_High_resolution_sunkeqiang_15/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_23",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_hand_in_left_eye/20180910_omron_eyelid_data_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_hand_in_left_eye/20180910_omron_eyelid_data_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_24",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_outside/20181127_ghostface_high_reso_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_outside/20181127_ghostface_high_reso_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_25",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_outside/20180823_High_resolution_sunkeqiang_12/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_outside/20180823_High_resolution_sunkeqiang_12/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_26",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_outside/20181203-close_range_high_pixel_face_data-liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_outside/20181203-close_range_high_pixel_face_data-liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_27",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_outside/20181213-middle_range_yaw_and_pitch_photo_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_outside/20181213-middle_range_yaw_and_pitch_photo_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_28",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_outside/20180821_High_resolution_sunkeqiang_15/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_outside/20180821_High_resolution_sunkeqiang_15/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_occ_29",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/roi_data/occ_data_outside/20180910_omron_eyelid_data_liutinghao/eyestate_label_from_landmark_and_occ_result_0.33.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/occ_data_outside/20180910_omron_eyelid_data_liutinghao/Images/",
                source="lustre",
            ),
            dict(
                dataset_name="iris_normal_0",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/glass_normal/gt_iris_cls_for_train.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/glass_normal/NewMeanPose/Images",
                source="lustre",
            ),
            dict(
                dataset_name="iris_normal_1",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/glass_normal1/gt_iris_cls_for_train.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/glass_normal1/NewMeanPose/Images",
                source="lustre",
            ),
            dict(
                dataset_name="iris_normal_2",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/glass_normal2/gt_iris_cls_for_train.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/glass_normal2/NewMeanPose/Images",
                source="lustre",
            ),
            dict(
                dataset_name="iris_normal_3",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/image_3/gt_iris_cls_for_train.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/image_3/NewMeanPose/Images",
                source="lustre",
            ),
            dict(
                dataset_name="iris_normal_4",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/image_6/gt_iris_cls_for_train.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/image_6/NewMeanPose/Images",
                source="lustre",
            ),
            dict(
                dataset_name="iris_normal_5",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/image_7/gt_iris_cls_for_train.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/image_7/NewMeanPose/Images",
                source="lustre",
            ),
            dict(
                dataset_name="iris_normal_6",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/image_8/gt_iris_cls_for_train.txt",
                image_rootpath="/mnt/lustre/hanxiaoyang/data/train_data/alignment/NewMeanPose/NewData/image_8/NewMeanPose/Images",
                source="lustre",
            ),
        ],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        image_size=dict(whole_size=384, center_size=256, expand_ratio=0.15),
        label_format="txt",
        data_infos=[
            dict(
                dataset_name="test_iris",
                repeats=1,
                json_file_list="/mnt/lustre/chenzukai/Data_t1/eyestate_Label/common_set_eyestate.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/face106pts/crop/common_set/Images/",
                source="lustre",
            ),
        ],
        pipeline=validate_pipeline,
    )
)
