# need to check outside var
crop_size = 112
landmark_num = 106
final_image_format_type = 'GRAY'

# dataset only var
dataset_type = "HeadCropDataset"
dataloader_type = "AvoidDeadLockDataLoader"

img_norm_cfg = dict(norm_type="z-score")


train_pipeline = [
    dict(type="HeadRandomAffineGetMat", crop_size=crop_size, expand_ratio=0.25, 
        degree=0, translate=0.3, scale_ratio=0.25),
    dict(type="HeadWarpAffineLabel"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="HeadRandomCrop", ratio=0.1, part_type='heads'),
    # dict(type="MotionBlur", Ls=[10, 20], probs=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="HeadLabelToTensor", keys=['rect']),
    dict(type="Collect", image_keys=['origin_image'],
         label_keys=['rect'],util_keys=['filename'])
]

validate_pipeline = [
    dict(type="HeadRandomAffineGetMat", crop_size=crop_size, expand_ratio=0.25, 
        degree=0, translate=0.3, scale_ratio=0.25),
    dict(type="HeadWarpAffineLabel"),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="HeadLabelToTensor", keys=['rect']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['rect'])
]

# lmdb config
train_db_info = dict(
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/head_refine_lmdb/",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info = dict(
    lmdb_path="/mnt/lustre/liutinghao/base_training_code/lmdb_path/sketch/head_refine_lmdb/",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

data = dict(
    train=dict(
        type=dataset_type,
        image_size=dict(whole_size=(384,384), center_size=256, expand_ratio=0.15),
        label_format="txt",
        data_infos=[
            dict(
                dataset_name='Head_BilibiliDance1',
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/refine_dataset/head_refine/Head_BilibiliDance1/Head_BilibiliDance1.txt",
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_BilibiliDance1/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ), 
            dict(
                dataset_name='Head_BilibiliDance2',
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/refine_dataset/head_refine/Head_BilibiliDance2/Head_BilibiliDance2.txt",
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_BilibiliDance2/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name='Head_Bilibili_liveshow_1',
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/refine_dataset/head_refine/Head_Bilibili_liveshow_1/Head_Bilibili_liveshow_1.txt",
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_Bilibili_liveshow_1/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name='Head_AllYaw',
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/refine_dataset/head_refine/Head_AllYaw/Head_AllYaw.txt",
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_AllYaw/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name='Head_DynamicOcclusion',
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/refine_dataset/head_refine/Head_DynamicOcclusion/Head_DynamicOcclusion.txt",
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_DynamicOcclusion/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name='Head_LargePitch',
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/refine_dataset/head_refine/Head_LargePitch/Head_LargePitch.txt",
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_LargePitch/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
        ],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        image_size=dict(whole_size=(384,384), center_size=256, expand_ratio=0.15),
        label_format="txt",
        data_infos=[
            dict(
                dataset_name='hardcase',
                repeats=1,
                json_file_list="/mnt/lustre/liutinghao/train_data/refine_dataset/head_refine/hardcase/hardcase.txt",
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Test/hardcase/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
        ],
        pipeline=validate_pipeline,
    )
)
