dataset_type="Face106ptLandmarkLMDBDataset"
img_norm_cfg = dict(mean=[],std=[],norm_type="z-score",to_gray=True)
gaussian_cfg = dict(mean=0.0,std=0.05)
final_train_image_size = 112
train_pipeline = [
    dict(type="RandomAffineGetMat",degrees=50,translate=0.3,expand_ratio=0.2,scale_ratio=0.3,input_size=final_train_image_size,flip_ratio=0.5),
    dict(type="ToTensor"),
    dict(type="WarpAffineWithLandmark",backend="gpu"),
    dict(type="MotionBlur",blur_LS=[40,80],blur_probs=0.4),
    dict(type="Normalize",**img_norm_cfg,backend="gpu"),
    dict(type="Collect",keys=['img','gt_face_106pt'])
]
validate_pipeline = [
    dict(type="LoadJsons",with_attribute=False),
    dict(type="LoadImages"),
    dict(type="RandomAffineGetMat",degrees=50,translate=0.3,expand_ratio=0.2,scale_ratio=0.3,input_size=final_train_image_size,flip_ratio=0.5),
    dict(type="ToTensor"),
    dict(type="WarpAffineWithLandmark",backend="gpu"),
    dict(type="MotionBlur",blur_LS=[40,80],blur_probs=0.4),
    dict(type="Normalize",**img_norm_cfg,backend="gpu"),
    dict(type="Collect",keys=['img','gt_face_106pt'])
]
test_pipeline = [
    dict(type="LoadJsons",with_attribute=False),
    dict(type="LoadImages"),
    dict(type="RandomAffineGetMat",input_size=final_train_image_size),
    dict(type="ToTensor"),
    dict(type="WarpAffineWithOutLandmark",backend="gpu"),
    dict(type="Normalize",**img_norm_cfg,backend="gpu"),
    dict(type="Collect",keys=['img'])
]
train_lmdb_path=""
test_lmdb_path=""
validate_lmdb_path=""

data = dict(
    train=dict(
        type=dataset_type,
        data=[
            dict(
            dataset_name="meitu_2016",
            json_file_list="",
            image_rootpath="",
            image_Source="ceph",
            ceph_path="sh40hdd",
            eyelid_version="20180930",
            eyebrow_version="20180930",
            nose_version="20180930",
            mouth_version="20180930",
            contour_version="20180930"),
        ],
        attribute_weight=dict(
            headpose=dict(
                yaw=dict(
                    less_than=50,
                    more_than=20,
                    weight=5
                )
            )
        ),
        pipeline=train_pipeline
    ),
    validate=dict(
        type=dataset_type,
        data=[
            dict(dataset_name="meitu_2016",
            json_file_list="",
            image_rootpath="",
            image_Source="ceph",
            ceph_path="sh40hdd",
            eyelid_version="20180930",
            eyebrow_version="20180930",
            nose_version="20180930",
            mouth_version="20180930",
            contour_version="20180930"),
        ],
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=2,metrix="nme",main_dataset_name="meitu_2016")
