# need to check outside var
crop_size = 64
landmark_num = 106
num_output = 2
final_image_format_type = 'GRAY'

# dataset only var
dataset_type = "EyeStateCropDataset"
dataloader_type = "DataLoader"

img_norm_cfg = dict(norm_type="z-score")

train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.0, degree=15, translate=0.100, scale_ratio=0.2),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="MotionBlur", Ls=[40, 75], probs=0.15),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_eyestate']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_eyestate'])
]

validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.0),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['gt_eyestate']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['gt_eyestate'])
]

# lmdb config
train_db_info = dict(
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/eyestate_v10/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info = dict(
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/eyestate_v10/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

# 3.7.8 datasets
data = dict(
    train=dict(
        type=dataset_type,
        image_size=dict(whole_size=120, center_size=64, expand_ratio=0.0),
        label_format='txt',
        data_infos=[
            dict(
                dataset_name="eye_normal_0",
                repeats=1,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200111_0112_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB_label/gt01_pure_for_train_rgb_allOpen.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200111_0112_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB_label/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_1",
                repeats=1,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0108_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB/gt01_pure_for_train_rgb_allOpen.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0108_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_2",
                repeats=1,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0115_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_IR/gt01_pure_for_train_rgb_allOpen.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0115_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_IR/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_3",
                repeats=1,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20190107_xiaomi_RGB_外国人睁闭眼_crop/gt01_pure_for_train_select_30percent_blur_invalid_close.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20190107_xiaomi_RGB_外国人睁闭眼_crop/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_4",
                repeats=1,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200302_LatinAmerica_Mexico_29_72642_AE_lilei/gt01_pure_for_train_rgb.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200302_LatinAmerica_Mexico_29_72642_AE_lilei/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_5",
                repeats=2,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200111_0112_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB_label/gt01_pure_for_train_rgb.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200111_0112_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB_label/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_6",
                repeats=2,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0108_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB/gt01_pure_for_train_rgb.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0108_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_RGB/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_9",
                repeats=1,
                json_file_list="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0115_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_IR/gt01_pure_for_train_rgb.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0115_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_IR/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_10",
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/20190107_xiaomi_RGB_外国人睁闭眼_crop/gt01_pure_for_train_select_30percent.txt",
                image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/20190107_xiaomi_RGB_外国人睁闭眼_crop/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_11",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/southeast/gt01_pure_for_train_rgb.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/southeast/Image_rgb/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_12",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/southeast/gt01_pure_for_train_ir.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/southeast/Image_ir/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_13",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/mideast/gt01_pure_for_train_rgb.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/mideast/Image_rgb/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_14",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/mideast/gt01_pure_for_train_ir.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/mideast/Image_ir/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_15",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/white/gt01_pure_for_train_rgb.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/white/Image_rgb/",
                source="lustre",
            ),
            dict(
                dataset_name="eye_normal_16",
                repeats=1,
                json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/white/gt01_pure_for_train_ir.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/white/Image_ir/",
                source="lustre",
            ),
            # dict(
            #     dataset_name="eye_normal_17",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_18",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal/gt01_hard_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_19",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_hard/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_hard/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_20",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_hard/gt01_hard_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_hard/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_21",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal1/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal1/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_22",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal1/gt01_hard_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal1/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_23",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark1/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark1/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_24",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark1/gt01_hard_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark1/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_25",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal2/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal2/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_26",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal2/gt01_hard_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_normal2/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_27",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark2/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark2/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_28",
            #     repeats=3,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark2/gt01_hard_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/glass_dark2/Images/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_53",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/ir3/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/ir3/image_3/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_54",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/ir4/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/ir4/image_4/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_55",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/rgb3/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/rgb3/image_3/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_56",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/rgb6/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/rgb6/image_6/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_57",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/lisiying1/data/eyestate/huawei/train/rgb78/gt01_pure_for_train.txt",
            #     image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/rgb78/image_78/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_58",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/faceblur_crop/train_faceblur_open.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/faceblur_crop/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_59",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/faceblur_crop/train_faceblur_close.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/faceblur_crop/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_60",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/faceblur_crop/train_faceblur_invalid.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/faceblur_crop/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_61",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/mark_occulusion_crop/train_open_20180110.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/mark_occulusion_crop/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_62",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/mark_occulusion_crop/train_close_20180110.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/mark_occulusion_crop/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_63",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/train_20180129_open.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_64",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/train_20180129_close.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_65",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/train_20180129_invalid.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_66",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/train_object_renew.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/object_occlusion/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_67",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/20180320_glass_close_crop_liutinghao/train_20180320_glass_close.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/20180320_glass_close_crop_liutinghao/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_68",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/vivo_dark_image_list_folder/20181203_vivo_dark_train_crop_liukeyi_final.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/20181124_vivo_dark_train_crop_liukeyi/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_69",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/vivo_dark_image_list_folder/20181130_vivo_hardcase_train_crop_lighten_liukeyi_all.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/20181130_vivo_hardcase_train_crop_liukeyi/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_70",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/vivo_dark_image_list_folder/20181208_vivo_darkbug_collect_train_crop.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_dataset/20181208_vivo_darkbug_collect_train_crop_liukeyi/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_71",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/xiaomi_E8_train_data_crop/20180723E8_红外_crop_lilei/20180723_xiaomi_E8_spot_train_crop_lilei.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/xiaomi_E8_train_data_crop/20180723E8_红外_crop_lilei/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_72",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/xiaomi_E8_train_data_crop/20180724-xiaomi-eyestate-crop-lilei/20180724_xiaomi_E8_glass_train_crop_lilei.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/xiaomi_E8_train_data_crop/20180724-xiaomi-eyestate-crop-lilei/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_73",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180806_E8_IR_reflect_crop_dark_liutinghao/mark_dark_1.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180806_E8_IR_reflect_crop_dark_liutinghao/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_74",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180806_E8_IR_reflect_crop_light_liutinghao/mark_light_1.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180806_E8_IR_reflect_crop_light_liutinghao/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_75",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180809_remark_trainset_crop_liutinghao/mark_dark_1.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180809_remark_trainset_crop_liutinghao/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_76",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180809_remark_trainset_crop_liutinghao/mark_light_1.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180809_remark_trainset_crop_liutinghao/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_77",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180803_IR_hardcase_crop_liutinghao/mark_dark_1.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180803_IR_hardcase_crop_liutinghao/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_78",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180803_IR_hardcase_crop_liutinghao/mark_light_1.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20180803_IR_hardcase_crop_liutinghao/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_79",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/IR_dataset_labels/20181214_E1_IR_reflect_glass_train_crop_label.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20181214_E1_IR_reflect_glass_train_crop/",
            #     source="lustre",
            # ),
            # dict(
            #     dataset_name="eye_normal_80",
            #     repeats=1,
            #     json_file_list="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/IR_dataset_labels/20181214_E8_IR_reflect_glass_train_crop_label.txt",
            #     image_rootpath="/mnt/lustre/liutinghao/liukeyi/data/eye_state_data/new_ir/20181214_E8_IR_reflect_glass_train_crop/",
            #     source="lustre",
            # ),
        ],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        image_size=dict(whole_size=120, center_size=64, expand_ratio=0.0),
        label_format="txt",
        data_infos=[
            dict(
                dataset_name="test_black",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/20200107_0115_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_IR/gt01_pure_for_train_rgb_allOpen.txt.test.txt",
                image_rootpath="/mnt/lustre/lilei/new/Data_t1/train_data/eye_train_data/20200107_0115_Eyestate_Black_Africa_HuaWeiTaurus_lisiying1_IR/Image/",
                source="lustre",
            ),
            dict(
                dataset_name="test_chinese",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/rgb3/gt01_pure_for_train.txt.test.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/rgb3/image_3/",
                source="lustre",
            ),
            dict(
                dataset_name="test_southeast",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/southeast/gt01_pure_for_train_rgb.txt.test.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/southeast/Image_rgb/",
                source="lustre",
            ),
            dict(
                dataset_name="test_white",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/white/gt01_pure_for_train_rgb.txt.test.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/white/Image_rgb/",
                source="lustre",
            ),
            dict(
                dataset_name="test_mideast",
                repeats=1,
                json_file_list="/mnt/lustre/hanxiaoyang/data/train_data/eye_train_data/mideast/gt01_pure_for_train_rgb.txt.test.txt",
                image_rootpath="/mnt/lustre/lisiying1/data/eyestate/huawei/train/mideast/Image_rgb/",
                source="lustre",
            ),
        ],
        pipeline=validate_pipeline,
    )
)
