
#variables
backbone_channels = [1, 4, 8, 16, 80]
model_landmark_num=106
model_crop_size = 63


#module
model = dict(
    type = 'TinyRotate_backbone_v1',
    crop_size = model_crop_size,
    landmark_num = model_landmark_num,
    feature_map_size = 4,
    BN_type = 'BatchNormal',
    Relu_type = 'Relu',
    init_relu = 'leaky_relu',
    init_type = 'kaiming_uniform',
    backbone=dict(
        type='StraightBackbone',
        conv_layer_number=3,
        layers=[dict(type='Conv2dBlock',   inchannel=backbone_channels[0], bias=True,
                    outchannel=backbone_channels[1], kernel_size=5, padding=0, stride=2),
                dict(type='Conv2dBlock', inchannel=backbone_channels[1], bias=True,
                    outchannel=backbone_channels[2], kernel_size=5, padding=0, stride=1),
                dict(type='MaxPool', kernel_size=3, stride=2, padding=0, ceil_mode=False),
                ]
    ),
    regression_head = dict(
        type='FCHead',
        conv_layer_number = 2,
        layers=[dict(type='Conv2dBlock', inchannel=backbone_channels[2], bias=True,
                    outchannel=backbone_channels[3], kernel_size=3, padding=1, stride=2),
                dict(type='Conv2dBlock',  inchannel=backbone_channels[3], bias=True,
                    outchannel=backbone_channels[4], kernel_size=3, padding=0, stride=1)],
        fc_number = 3,
        fc_channels = [64, 64, 20],
    ),
)

losses = dict(
    losses_info = [
        dict(type='CERotate',
            loss_weight=1.0,
            name='coarse_roll'),
        ]
)
metrics = dict(
    metrics_info = [
        dict(type='CERotate',
            metric_weight=1.0,
            name='coarse_roll'),
        ]
)
