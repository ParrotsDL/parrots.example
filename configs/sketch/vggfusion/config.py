_base_=['../configs/base106/base106.py']

backbone_channels = [1, 10, 16, 24, 32, 64, 64, 96, 128, 128]
cls_head_channels = 40
occ_head_channels = 40
model_landmark_num=106
model_crop_size = 112

model = dict(
    type = 'TinyAlign106_backbone_v1',
    crop_size = model_crop_size,
    landmark_num = model_landmark_num,
    feature_map_size = 4,
    BN_type = 'BatchNormal',
    Relu_type = 'PRelu',
    init_relu = 'leaky_relu',
    init_type = 'kaiming_uniform',
    backbone=dict(
        type='FusionBackbone',
        conv_layer_number=12,
        layers1=[dict(type='Conv2dBlock',   inchannel=backbone_channels[0],  
                     outchannel=backbone_channels[1], kernel_size=5, padding=0, stride=2), # 54
                 dict(type='AvgPool', kernel_size=2, stride=2, padding=0),  # 27
                 dict(type='Conv2dBlock', inchannel=backbone_channels[1],
                     outchannel=backbone_channels[2], kernel_size=3, padding=0, stride=1), # 25
                 dict(type='Conv2dBlock', inchannel=backbone_channels[2],
                     outchannel=backbone_channels[3], kernel_size=3, padding=0, stride=1),  # 23
                 dict(type='Conv2dBlock',   inchannel=backbone_channels[3],
                     outchannel=backbone_channels[4], kernel_size=3, padding=1, stride=2)], # 12
 
        layers2=[dict(type='Conv2dBlock', inchannel=backbone_channels[4],
                     outchannel=backbone_channels[5], kernel_size=3, padding=0, stride=1),
                 dict(type='Conv2dBlock', inchannel=backbone_channels[5],
                     outchannel=backbone_channels[6], kernel_size=3, padding=0, stride=1)], # 8
 
        layers3=[dict(type='Conv2dBlock', inchannel=backbone_channels[6],
                     outchannel=backbone_channels[7], kernel_size=3, padding=1, stride=2),
                 dict(type='Conv2dBlock', inchannel=backbone_channels[7],
                     outchannel=backbone_channels[8], kernel_size=3, padding=1, stride=1)],

        layers1_out=[dict(type='Conv2dBlock', inchannel=backbone_channels[4],
                         outchannel=backbone_channels[4], kernel_size=3, padding=0, stride=1),  # 10
                     dict(type='Conv2dBlock', inchannel=backbone_channels[4],
                         outchannel=backbone_channels[4], kernel_size=3, padding=0, stride=2)],  # 4
        layers2_out=[dict(type='Conv2dBlock', inchannel=backbone_channels[6],
                         outchannel=backbone_channels[6], kernel_size=3, padding=1, stride=2)],

        layers_fusion=[dict(type='Conv2dBlock', inchannel=backbone_channels[4]+backbone_channels[6]+backbone_channels[8],
                            outchannel=backbone_channels[4]+backbone_channels[6]+backbone_channels[8], kernel_size=1, padding=0, stride=1)]
    ),
    regression_head = dict(
        type='FCHead',
        conv_layer_number = 1,
        layers=[dict(type='Conv2dBlock',  inchannel=backbone_channels[4]+backbone_channels[6]+backbone_channels[8],
                    outchannel=backbone_channels[9], kernel_size=3, padding=1, stride=1)],
        fc_number = 3,
        fc_channels = [128, 128, model_landmark_num*2],
    ),
    cls_head=dict(
        type='FCHead',
        conv_layer_number=1,
        layers=[dict(type='Conv2dBlock',  inchannel=backbone_channels[4]+backbone_channels[6]+backbone_channels[8],
                    outchannel=cls_head_channels, kernel_size=3, padding=1, stride=1)],
        fc_number=3,
        fc_channels=[128, 128, 2],
    ),
    occ_head=dict(
        type='FCHead',
        conv_layer_number=1,
        layers=[dict(type='Conv2dBlock', inchannel=backbone_channels[4]+backbone_channels[6]+backbone_channels[8],
                    outchannel=occ_head_channels, kernel_size=3, padding=1, stride=1)],
        fc_number=3,
        fc_channels=[128, 128, model_landmark_num*2],
    ),
)

