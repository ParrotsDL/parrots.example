# model config
backbone_channels = [1, 10, 16, 16, 24, 32, 32, 48, 64]
model_landmark_num=106
cls_head_channels=32
model_crop_size = 192
rect_num = 5

#module
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
        conv_layer_number=9,
        layers=[dict(type="InterpolateLayer",output_size=[112,112]),
                dict(type='Conv2dBlock',   inchannel=backbone_channels[0],
                    outchannel=backbone_channels[1], kernel_size=5, padding=0, stride=2),
                dict(type='AvgPool', kernel_size=2, stride=2, padding=0),
                dict(type='Conv2dBlock', inchannel=backbone_channels[1],
                    outchannel=backbone_channels[2], kernel_size=3, padding=0, stride=1),
                dict(type='Conv2dBlock', inchannel=backbone_channels[2],
                    outchannel=backbone_channels[3], kernel_size=3, padding=0, stride=1),
                dict(type='RepVGGBlock',   inchannel=backbone_channels[3],
                    outchannel=backbone_channels[4], kernel_size=3, padding=1, stride=2),
                dict(type='Conv2dBlock', inchannel=backbone_channels[4],
                    outchannel=backbone_channels[5], kernel_size=3, padding=0, stride=1),
                dict(type='Conv2dBlock', inchannel=backbone_channels[5],
                    outchannel=backbone_channels[6], kernel_size=3, padding=0, stride=1),
                dict(type='RepVGGBlock',   inchannel=backbone_channels[6],
                    outchannel=backbone_channels[7], kernel_size=3, padding=1, stride=2)],
    ),
    regression_head = dict(
        type='FCHead',
        conv_layer_number = 1,
        layers=[dict(type='RepVGGBlock',  inchannel=backbone_channels[7],
                    outchannel=backbone_channels[8], kernel_size=3, padding=1, stride=1)],
        fc_number = 3,
        fc_channels = [128, 128, model_landmark_num*2+rect_num*4],
    ),
    cls_head=dict(
        type='FCHead',
        conv_layer_number=1,
        layers=[dict(type='RepVGGBlock',  inchannel=backbone_channels[7],
                    outchannel=cls_head_channels, kernel_size=3, padding=1, stride=1)],
        fc_number=3,
        fc_channels=[64, 64, 1],
    ),
)


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
