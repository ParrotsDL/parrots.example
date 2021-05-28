# outside var
# nme_type = 'STNormDis'

# need to check outside var

crop_size = 63
landmark_num = 106
final_image_format_type = 'GRAY'
rotate_num=20

# dataset only var
dataset_type="HeadPoseFromMeshDataset"
dataloader_type="DataLoader"

img_norm_cfg = dict(norm_type="z-score")
train_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.5, 
        degree=180,translate=0.1, scale_ratio=0.06857),
    dict(type="GetRotate", label_type='rot_gt',region_type='head'),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="RandomOcc", ratio=0.1,part_type='heads'),
    dict(type='GaussianNoise', sigma=3.0),
    # dict(type="MotionBlur", Ls=[10, 20], probs=0.2),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['rot_gt']),
    dict(type="Collect", image_keys=['origin_image'],
         label_keys=['rot_gt'],util_keys=['filename'])
]
validate_pipeline = [
    dict(type="RandomAffineGetMat", crop_size=crop_size, expand_ratio=0.5, degree=180),
    dict(type="GetRotate", label_type='rot_gt',region_type='head'),
    dict(type="ColorConvert", final_image_format_type=final_image_format_type),
    dict(type="WarpAffineImage"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=['origin_image']),
    dict(type="LabelToTensor", keys=['rot_gt']),
    dict(type="Collect", image_keys=['origin_image'], label_keys=['rot_gt'])
]

data = dict(
        train=dict(
            type=dataset_type,
            pipeline = train_pipeline),
        val = dict(
            type=dataset_type,
            pipeline = validate_pipeline),
        )

# test_pipeline = [
#     dict(type="RandomAffineGetMat",input_size=crop_size),
#     dict(type="ToTensor"),
#     dict(type="WarpAffineImage",backend="gpu"),
#     dict(type="Normalize", **img_norm_cfg,backend="gpu"),
#     dict(type="Collect", keys=['img'])
# ]

# lmdb config
train_db_info=dict(
    lmdb_path="/mnt/lustre/wangchao6/base_training_code/lmdb_path/sketch/subdataset_head_pose_from_mesh",
    lmdb_name="Train",
    image_type_in_lmdb="path"
)

validate_db_info=dict(
    lmdb_path="/mnt/lustre/wangchao6/base_training_code/lmdb_path/sketch/subdataset_head_pose_from_mesh",
    lmdb_name="Validate",
    image_type_in_lmdb="path"
)

data = dict(
    train=dict(
        type=dataset_type,
        image_size=dict(whole_size=(384,384), center_size=256, expand_ratio=0.15),
        label_format="json",
        data_infos=[
            dict(
                dataset_name="back_head_img_train",
                repeats=0.5,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_img_train.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
            dict(
                dataset_name="back_head_3D_exp_img_train",
                repeats=2.5,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_3D_exp_im_train.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
            dict(
                dataset_name="back_head_3D_exp_img_train1",
                repeats=2,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_3D_exp_im_train1.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
            dict(
                dataset_name="back_head_3D_img_train",
                repeats=4,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_3D_im_train.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
            dict(
                dataset_name="Head_BilibiliDance2",
                repeats=0.45,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/Head_BilibiliDance2.txt",
                json_rootpath='/mnt/lustre/wangkun1/data_t1/data_process_wangchao6/inference_face360/save_test/',
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_BilibiliDance2/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name="Head_AllYaw",
                repeats=0.4,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/Head_AllYaw.txt",
                json_rootpath='/mnt/lustre/wangkun1/data_t1/data_process_wangchao6/inference_face360/save_test/',
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_AllYaw/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name="Head_DynamicOcclusion",
                repeats=1,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/Head_DynamicOcclusion.txt",
                json_rootpath='/mnt/lustre/wangkun1/data_t1/data_process_wangchao6/inference_face360/save_test/',
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_DynamicOcclusion/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name="Head_LargePitch",
                repeats=5,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/Head_LargePitch.txt",
                json_rootpath='/mnt/lustre/wangkun1/data_t1/data_process_wangchao6/inference_face360/save_test/',
                image_rootpath="s3://ARFace.Refine_Bucket/Head_Refine/Train/Head_LargePitch/Image/",
                source="ceph",
                ceph_clustre="sh40_ssd"
            ),
            dict(
                dataset_name="AllYaw_girl",
                repeats=1,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/AllYaw_girl.txt",
                json_rootpath='/mnt/lustre/wangkun1/data_t1/data_process_wangchao6/inference_face360/save_test/',
                image_rootpath="/mnt/lustre/wangchao6/datasets/head_refine",
                source="lustre",
            ),
            dict(
                dataset_name="clean_v2data",
                repeats=0.1,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/clean_v2data.txt",
                json_rootpath='',
                image_rootpath="",
                source="lustre",
            ),
            dict(
                dataset_name="AllPose_Train",
                repeats=3,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/AllPose_mesh_train.txt",
                json_rootpath='/mnt/lustre/wangkun1/data_t1/data_process_wangchao6/inference_face360/save_test/',
                image_rootpath="/mnt/lustre/wangchao6/datasets/head_refine",
                source="lustre",
            ),
        ],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        image_size=dict(whole_size=(384,384), center_size=256, expand_ratio=0.15),
        label_format="json",
        data_infos=[
            dict(
                dataset_name="back_head_img_test",
                repeats=1,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_img_test.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
            dict(
                dataset_name="back_head_3D_exp_img_test",
                repeats=1,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_3D_exp_im_test.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
            dict(
                dataset_name="back_head_3D_exp_img_test1",
                repeats=1,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_3D_exp_im_test1.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
            dict(
                dataset_name="back_head_3D_img_test",
                repeats=1,
                json_file_list="/mnt/lustre/wangchao6/datasets/pose/head/back_head_data/headpose_txt_from_mesh/back_head_3D_im_test.txt",
                image_rootpath="/mnt/lustre/wangchao6/datasets/pose/head/",
                json_rootpath='/mnt/lustre/share/wangkun1/',
                source="lustre",
            ),
        ],
        pipeline=validate_pipeline,
    )
)

# evaluation = dict(interval=2,metrix="nme",main_dataset_name="meitu_2016")
