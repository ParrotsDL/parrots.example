# variables
backbone_channels = [1, 8, 8, 16, 16, 48, 80, 80, 128, 128, 256]
cls_head_channels = 40
occ_head_channels = 40
model_crop_size = 192
model_num_output = 2

# module
model = dict(
    type='ResNetEyeIrisState',
    crop_size=model_crop_size,
    feature_map_size=1,
    BN_type='BatchNormal',
    Relu_type='Relu',
    init_relu='leaky_relu',
    init_type='kaiming_uniform',
    backbone=dict(
        type='StraightBackbone',
        conv_layer_number=10,
        layers=[dict(type='Conv2dBlock', inchannel=backbone_channels[0],
                     outchannel=backbone_channels[1], kernel_size=3, padding=1, stride=2),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[1],
                     outchannel=backbone_channels[2], kernel_size=3, branch=1, stride=1),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[2],
                     outchannel=backbone_channels[3], kernel_size=3, branch=2, stride=2),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[3],
                     outchannel=backbone_channels[4], kernel_size=3, branch=2, stride=2),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[4],
                     outchannel=backbone_channels[5], kernel_size=3, branch=2, stride=1),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[5],
                     outchannel=backbone_channels[6], kernel_size=3, branch=2, stride=2),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[6],
                     outchannel=backbone_channels[7], kernel_size=3, branch=2, stride=2),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[7],
                     outchannel=backbone_channels[8], kernel_size=3, branch=2, stride=1),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[8],
                     outchannel=backbone_channels[9], kernel_size=3, branch=2, stride=2),
                dict(type='ResnetBlock_v2', inchannel=backbone_channels[9],
                     outchannel=backbone_channels[10], kernel_size=3, branch=2, stride=1),
                ]
    ),
    regression_head=dict(
        type='FCHead',
        conv_layer_number=0,
        layers=[dict(type='MaxPool', kernel_size=4, stride=4, outchannel=backbone_channels[-1]),
                ],
        fc_number=3,
        fc_channels=[128, 128, model_num_output * 2],
    ),
)

losses = dict(
    losses_info = [  
        dict(type = 'EyeStateLoss', 
             base_loss=dict(type="CrossEntropyLoss"),
             eyestate_features=["left_eyestate","right_eyestate"],
             weight = 1)
    ]
)

metrics = dict(
    metrics_info = [
        dict(type="EyeStateMetric",
            base_metric=dict(type="DichotomyWithThresholdMetric",threshold=0.5),
            eyestate_feature='left_eyestate'
        ),
        dict(type="EyeStateMetric",
            base_metric=dict(type="DichotomyWithThresholdMetric",threshold=0.5),
            eyestate_feature='right_eyestate'
        ),
    ]
)


