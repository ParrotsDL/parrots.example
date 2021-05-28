backbone_channels = [1, 10, 16, 24, 32, 64, 64, 96, 128, 128]
model_landmark_num=106
cls_head_channels=80
model_crop_size = 112

model = dict(
    type = 'TinyAlign106_backbone_v1',
    crop_size = model_crop_size,
    landmark_num = model_landmark_num,
    feature_map_size = 4,
    BN_type = 'BN',
    Relu_type = 'PReLU',
    init_relu = 'leaky_relu',
    init_type = 'kaiming_uniform',
    backbone=dict(
        type='StraightBackbone',
        conv_layer_number=12,
        layers=[dict(type='Conv2dBlock',   inchannel=backbone_channels[0],
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
        fc_channels = [128, 128, model_landmark_num*2],
    ),
    cls_head = dict(
        type='FCHead',
        conv_layer_number = 2,
        layers=[dict(type='DWConv2dBlock',  inchannel=backbone_channels[8],
                    outchannel=cls_head_channels, kernel_size=3, padding=1, stride=1),
                dict(type='SELayer',
                    outchannel=cls_head_channels, feature_size=4, reduction=4)],
        fc_number = 3,
        fc_channels = [64, 64, 1],
    ),
    # cls_head=dict(
    #     type='FCHead',
    #     conv_layer_number=1,
    #     layers=[dict(type='DWConv2dBlock',  inchannel=backbone_channels[8],
    #                 outchannel=cls_head_channels, kernel_size=3, padding=1, stride=1),
    #             dict(type='SELayer',
    #                 outchannel=cls_head_channels, feature_size=4, reduction=4)],
    #     fc_number=3,
    #     fc_channels=[64, 64, 2],
    # )
)

# loss
losses = dict(
    losses_info = [  
        dict(type = 'QualityJudgeLoss', 
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             nme_threshold=0.5,
             branch = 'main',
             part = ['face_106pt_all'],
             weight = 10)    
    ]
)


metrics = dict(
    metrics_info = [
        dict(type="LandmarkMetric",
             base_metric=dict(type="BaseNMEMetric"),
             branch=["main"],
             part=["face_106pt_all"],
             metric_name="face_106",
        ),
        dict(type="NMEQualityAccMetric",
             base_metric=dict(type="SmoothL1LossWithThreshold"),
             branch="main",
             nme_threshold=0.5,
             part=['face_106pt_all']
        )
    ]
)
