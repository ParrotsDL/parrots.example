
# model config
backbone_channels = [1, 6, 16, 16, 32, 40, 48, 80, 96]
cls_head_channels = 64
model_landmark_num=106
model_crop_size = 112

#module
model = dict(
    type = 'TinyRect106_backbone_v1',
    crop_size = model_crop_size,
    landmark_num = model_landmark_num,
    feature_map_size = 4,
    BN_type = 'BN',
    Relu_type = 'PReLU',
    init_relu = 'leaky_relu',
    init_type = 'kaiming_uniform',
    backbone=dict(
        type='StraightBackbone',
        conv_layer_number=8,
        layers=[dict(type='Conv2dBlock',   inchannel=backbone_channels[0],
                    outchannel=backbone_channels[1], kernel_size=5, padding=0, stride=2),
                dict(type='AvgPool', kernel_size=2, stride=2, padding=0),
                dict(type='Conv2dBlock', inchannel=backbone_channels[1],
                    outchannel=backbone_channels[2], kernel_size=3, padding=0, stride=1),
                dict(type='Conv2dBlock', inchannel=backbone_channels[2],
                    outchannel=backbone_channels[3], kernel_size=3, padding=0, stride=1),
                dict(type='Conv2dBlock',   inchannel=backbone_channels[3],
                    outchannel=backbone_channels[4], kernel_size=3, padding=1, stride=2),
                dict(type='Conv2dBlock', inchannel=backbone_channels[4],
                    outchannel=backbone_channels[5], kernel_size=3, padding=0, stride=1),
                dict(type='Conv2dBlock', inchannel=backbone_channels[5],
                    outchannel=backbone_channels[6], kernel_size=3, padding=0, stride=1),
                dict(type='Conv2dBlock',   inchannel=backbone_channels[6],
                    outchannel=backbone_channels[7], kernel_size=3, padding=1, stride=2)],
    ),
    regression_head = dict(
        type='FCHead',
        conv_layer_number = 1,
        layers=[dict(type='Conv2dBlock',  inchannel=backbone_channels[7],
                    outchannel=backbone_channels[8], kernel_size=3, padding=1, stride=1)],
        fc_number = 3,
        fc_channels = [128, 128, 4],
    ),
    cls_head=dict(
        type='FCHead',
        conv_layer_number=1,
        layers=[dict(type='Conv2dBlock',  inchannel=backbone_channels[7],
                    outchannel=cls_head_channels, kernel_size=3, padding=1, stride=1)],
        fc_number=3,
        fc_channels=[64, 64, 1],
    ),
)

losses = dict(
    losses_info = [
        dict(type='BBoxScoreLoss',
            base_loss=dict(type='SmoothL1LossWithThreshold'),
            loss_weight=75.0,
            name='score'),
        ]
)
metrics = dict(
    metrics_info = [
        dict(type='BBoxPointMetric',
            base_metric=dict(type='L1Metric'),
            metric_weight=1.0,
            name='point'),
        dict(type='BBoxScoreMetric',
            base_metric=dict(type='L1Metric'),
            metric_weight=1.0,
            name='score'),
        ]
)

