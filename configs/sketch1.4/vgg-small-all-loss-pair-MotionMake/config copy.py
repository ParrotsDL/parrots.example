_base_ = [
    '../../../configs/face_alignment/facial_landmark_main/models/vgg_small.py',
    '../../../configs/face_alignment/facial_landmark_main/datasets/NewMeanPose_v3_MotionMake_pair_multi_dataloader.py',
    '../../../configs/face_alignment/facial_landmark_main/schedules/schedule_v3.py',
]

samples_per_gpu = [64, 32]

model = dict(
    type='TinyAlign106_backbone_pair_hanxy',
)

# loss
losses = dict(
    losses_info=[
        # main loss
        dict(type='LandmarkLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch='main',
             part=['eyebrow_9pt', 'eyelid_10pt', 'nose_15pt', 'lip_20pt'],
             weight=73,
             loss_name="face_organs"),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch='main',
             part=['contour_33pt'],
             special_feature="Curve",
             weight=15,
             loss_name="face_contour"),
        # cf loss
        dict(type="ContourFittingLoss",
             base_loss=dict(type="CF_SoftWingLoss"),
             weight=30),
        # 3d loss
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['nose_15pt'],
             special_feature="Static3D",
             weight=2),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['eyebrow_9pt'],
             special_feature="Static3D",
             weight=12),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['lip_20pt'],
             special_feature="Static3D",
             weight=24),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['contour_33pt'],
             special_feature="Static3D",
             weight=12),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch="main",
             part=['eyelid_10pt'],
             special_feature="Static3D",
             weight=20),
        # shift loss
        dict(type='ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             weight=1.5 * 0.7),
        dict(type='ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             suffix="pair",
             weight=1.5 * 0.3),
        # pair loss
        dict(type='PairLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             weight=1.5),
        # diff loss
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['left_eyelid_10pt'],
             weight=15,
             loss_name="left_eye_diff"),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['right_eyelid_10pt'],
             weight=15,
             loss_name="right_eye_diff"),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['lip_20pt'],
             diff_part="inner",
             weight=10,
             loss_name="inner_lip_diff"),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['lip_20pt'],
             weight=5,
             loss_name="lip_diff")
    ]
)

dist_params = dict(backend='nccl', port=80215)

lr_config = dict(
    gamma=0.5,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[150, 180, 210, 240, 270])
total_epochs = 300
