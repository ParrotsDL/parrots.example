
#variables
backbone_channels = [1, 8, 16, 24, 32, 40, 64, 96, 128, 128]
cls_head_channels = 40
occ_head_channels = 40
model_landmark_num=68
model_crop_size = 112

#module
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
        type='StraightBackbone',
        conv_layer_number=9,
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
                    outchannel=backbone_channels[7], kernel_size=3, padding=1, stride=2),
                dict(type='Conv2dBlock', inchannel=backbone_channels[7],
                    outchannel=backbone_channels[8], kernel_size=3, padding=1, stride=1)],
    ),
    regression_head = dict(
        type='FCHead',
        conv_layer_number = 1,
        layers=[dict(type='Conv2dBlock',  inchannel=backbone_channels[8],
                    outchannel=backbone_channels[9], kernel_size=3, padding=1, stride=1)],
        fc_number = 3,
        fc_channels = [128, 128, model_landmark_num*2],
    ),
    cls_head=dict(
        type='FCHead',
        conv_layer_number=1,
        layers=[dict(type='Conv2dBlock',  inchannel=backbone_channels[8],
                    outchannel=cls_head_channels, kernel_size=3, padding=1, stride=1)],
        fc_number=3,
        fc_channels=[128, 128, 2],
    ),
    occ_head=dict(
        type='FCHead',
        conv_layer_number=1,
        layers=[dict(type='Conv2dBlock', inchannel=backbone_channels[8],
                    outchannel=occ_head_channels, kernel_size=3, padding=1, stride=1)],
        fc_number=3,
        fc_channels=[128, 128, model_landmark_num*2],
    ),
)


losses = dict(
    alignment = dict(
        type = 'face-alignment-loss',
        losses = [dict(type ='softwing_loss', thres1 = 1, thres2 = 10, curvature=0.3, name = 'face_loss', weight=106),
                  dict(type ='smoothl1_with_threshold', name = 'eyedif_left_loss', weight=10),
                  dict(type ='smoothl1_with_threshold', name = 'eyedif_right_loss', weight=10),
                  dict(type ='smoothl1_with_threshold', name = 'lipdif_loss', weight=5),
                  dict(type ='softwing_loss', thres1 = 1, thres2 = 10, curvature=0.3, name = '3d_loss', weight=60),
                ]
    ),
    cls = dict(
        type = 'face-cls-loss',
        losses = [dict(type = 'CrossEntropyLoss', name = 'cls_loss', thres=0.5, weight=1)],
    ),

    occ = dict(
        type = 'face_occ_loss',
        losses = [dict(type = 'CrossEntropyLoss', name = 'occ_loss', weight=1)],
    ),
)
metrics = dict(
        metric_alignment = [dict(type = 'nme', name='face_106')],
        metric_cls = [dict(type = 'nme', name='face_106'),
                      dict(type = 'accuracy', name='cls', thres=0.5)],
        metric_occ = [dict(type = 'nme', name='face_106'),
                      dict(type = 'accuracy', name='occ')],
)
