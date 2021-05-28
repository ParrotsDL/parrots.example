backbone_channels = [1, 10, 16, 24, 32, 64, 64, 96, 128, 128]
model_landmark_num=106
model_crop_size = 192
rect_num = 5

model = dict(
    type = 'TinyAlign106_with_rect',
    crop_size = model_crop_size,
    landmark_num = model_landmark_num,
    feature_map_size = 4,
    rect_num = rect_num,
    BN_type = 'BN',
    Relu_type = 'PReLU',
    init_relu = 'leaky_relu',
    init_type = 'kaiming_uniform',
    backbone=dict(
        type='StraightBackbone',
        conv_layer_number=13,
        layers=[dict(type="InterpolateLayer",output_size=[112,112]),
                dict(type='Conv2dBlock',   inchannel=backbone_channels[0],
                    outchannel=backbone_channels[1], kernel_size=5, padding=0, stride=2),
                dict(type='AvgPool', kernel_size=2, stride=2, padding=0),
                dict(type='DWConv2dBlock', inchannel=backbone_channels[1],
                    outchannel=backbone_channels[2], kernel_size=3, padding=0, stride=1),
                dict(type='DWConv2dBlock', inchannel=backbone_channels[2],
                    outchannel=backbone_channels[3], kernel_size=3, padding=0, stride=1),
                dict(type='Conv2dBlock',   inchannel=backbone_channels[3],
                    outchannel=backbone_channels[4], kernel_size=3, padding=1, stride=2),
                dict(type='DWConv2dBlock', inchannel=backbone_channels[4],
                    outchannel=backbone_channels[5], kernel_size=3, padding=0, stride=1),
                dict(type='DWConv2dBlock', inchannel=backbone_channels[5],
                    outchannel=backbone_channels[6], kernel_size=3, padding=0, stride=1),
                dict(type='SELayer',
                    outchannel=backbone_channels[6],feature_size=8, reduction=4),
                dict(type='Conv2dBlock',   inchannel=backbone_channels[6],
                    outchannel=backbone_channels[7], kernel_size=3, padding=1, stride=2),
                dict(type='SELayer',
                    outchannel=backbone_channels[7],feature_size=4, reduction=4),
                dict(type='DWConv2dBlock', inchannel=backbone_channels[7],
                    outchannel=backbone_channels[8], kernel_size=3, padding=1, stride=1),
                dict(type='SELayer',
                    outchannel=backbone_channels[8],feature_size=4, reduction=4)]
    ),
    regression_head = dict(
        type='FCHead',
        conv_layer_number = 2,
        layers=[dict(type='DWConv2dBlock',  inchannel=backbone_channels[8],
                    outchannel=backbone_channels[9], kernel_size=3, padding=1, stride=1),
                dict(type='SELayer',
                    outchannel=backbone_channels[9], feature_size=4, reduction=4)],
        fc_number = 3,
        fc_channels = [128, 128, model_landmark_num*2+rect_num*4],
    )
)

# loss
losses = dict(
    losses_info = [
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss"),
             branch = 'main',
             part = ['face_106pt_all'],
             weight = 106),
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "main",
             part = ['face_106pt_all'],
             weight = 1.2),
        dict(type = "ContourFittingLoss",
             base_loss=dict(type="CF_SoftWingLoss"),
             weight = 60),
        dict(type = "RectLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             rects = ['mouth','eye','eyebrow'],
             weight = 1.5,
             loss_name="all_rect"),
        dict(type = "RectShiftLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             rects = ['mouth','eye','eyebrow'],
             weight=1.2,
             loss_name="all_rect_shift"),
        dict(type = "DiffLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "main",
             part = ['lip_20pt'],
             weight = 5,
             loss_name="lip_diff")
    ]
)


metrics = dict(
    metrics_info = [
        dict(type = "RectMetric",
             base_metric=dict(type="BaseNMEMetric"),
             rects = ['mouth','eye','eyebrow']),
        dict(type = "ContourFittingMetric",
             base_metric=dict(type="ContourNMEMetric")),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["main"],
            part=["face_106pt_all"],
            metric_name="face_106_nme")
    ]
)
