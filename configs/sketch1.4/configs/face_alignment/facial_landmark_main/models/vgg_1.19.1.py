# model config
backbone_channels = [1, 8, 32, 32, 64, 64, 160, 160]
model_landmark_num = 106
model_crop_size = 112

# module
model = dict(
    type='TinyAlign106_backbone_v1',
    crop_size=model_crop_size,
    landmark_num=model_landmark_num,
    feature_map_size=4,
    BN_type='BN',
    Relu_type='PReLU',
    init_relu='leaky_relu',
    init_type='kaiming_uniform',
    backbone=dict(
        type='StraightBackbone',
        conv_layer_number=9,
        layers=[dict(type='Conv2dBlock', inchannel=backbone_channels[0],
                     outchannel=backbone_channels[1], kernel_size=5, padding=0, stride=2),  # 54
                dict(type='AvgPool', kernel_size=2, stride=2, padding=0),  # 27
                dict(type='Conv2dBlock', inchannel=backbone_channels[1],
                     outchannel=backbone_channels[2], kernel_size=3, padding=0, stride=1),  # 25
                dict(type='Conv2dBlock', inchannel=backbone_channels[2],
                     outchannel=backbone_channels[3], kernel_size=3, padding=0, stride=1),  # 23
                dict(type='AvgPool', kernel_size=2, stride=2, padding=0),  # 12
                dict(type='Conv2dBlock', inchannel=backbone_channels[3],
                     outchannel=backbone_channels[4], kernel_size=3, padding=0, stride=1),  # 10
                dict(type='Conv2dBlock', inchannel=backbone_channels[4],
                     outchannel=backbone_channels[5], kernel_size=3, padding=0, stride=1),  # 8
                dict(type='AvgPool', kernel_size=2, stride=2, padding=0),  # 4
                dict(type='Conv2dBlock', inchannel=backbone_channels[5],
                     outchannel=backbone_channels[6], kernel_size=3, padding=1, stride=1),  # 4
                ],
    ),
    regression_head=dict(
        type='FCHead',
        conv_layer_number=1,
        layers=[dict(type='Conv2dBlock', inchannel=backbone_channels[6],
                     outchannel=backbone_channels[7], kernel_size=3, padding=1, stride=1)],
        fc_number=3,
        fc_channels=[128, 128, model_landmark_num * 2],
    )
)

# loss
losses = dict(
    losses_info=[
        dict(type='LandmarkLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch='main',
             part=['face_106pt_all'],
             weight=1,
             loss_name="face_106"),
        dict(type='SpecialPointLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             special_feature="Static3D",
             weight=3),
        dict(type='ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['face_106pt_all'],
             weight=1),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['eyelid_10pt'],
             weight=10,
             loss_name="eye_diff"),
        dict(type="DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             part=['lip_20pt'],
             weight=5,
             loss_name="lip_diff"),
    ]
)

metrics = dict(
    metrics_info=[
        dict(type="LandmarkMetric",
             base_metric=dict(type="BaseNMEMetric"),
             branch=["main"],
             part=["face_106pt_all"],
             metric_name="face_106_nme")
    ]
)
