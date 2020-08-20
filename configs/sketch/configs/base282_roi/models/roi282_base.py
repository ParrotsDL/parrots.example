
#variables
mb_backbone_channels = [1, 8, 16, 16, 24, 32, 48, 64, 80]
lb_backbone_channels = [1, 8, 16, 24, 32, 32, 48, 64, 80]
eb_backbone_channels = [1, 8, 16, 16, 24, 24, 32, 48, 64]
ebb_backbone_channels = [1, 8, 8, 8, 16, 16, 32, 32]
cls_head_channels = 40
occ_head_channels = 40
face_landmark_num = 106
rect_num = 5
mouth_landmark_num = 64+20
eye_landmark_num = 24+19+10
eyebrow_landmark_num = 13+9
model_crop_size = 256  
# just for test on 106, need remove
backbone_channels = [1, 8, 16, 24, 32, 40, 64, 96, 128, 128]
model_landmark_num=106
model_crop_size = 112

#module
model = dict(
    type = 'Align282_Roi_v1',
    crop_size = model_crop_size,
    rect_num = rect_num,
    landmark_num = face_landmark_num+mouth_landmark_num+eye_landmark_num*2+eyebrow_landmark_num*2,
    BN_type = 'BatchNormal',
    Relu_type = 'PRelu',
    init_relu = 'leaky_relu',
    init_type = 'kaiming_uniform',
    # main 106
    main_branch=dict(
        type='Align282MainBranch_v1',
        feature_map_size = 4,
        backbone1=dict(
            type='StraightBackbone',
            conv_layer_number=2,
            layers=[dict(type='InterpolateLayer', output_size=[128, 128]),
                    dict(type='Conv2dBlock',   inchannel=mb_backbone_channels[0],
                    outchannel=mb_backbone_channels[1], kernel_size=5, padding=2, stride=2)],
        ),
        backbone2=dict(
            type='StraightBackbone',
            conv_layer_number=7,
            layers=[dict(type='AvgPool', kernel_size=2, stride=2, padding=0), # 32
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[1],
                         outchannel=mb_backbone_channels[2], kernel_size=3, padding=0, stride=1), # 30
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[2],
                         outchannel=mb_backbone_channels[3], kernel_size=3, padding=1, stride=2), # 15
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[3],
                         outchannel=mb_backbone_channels[4], kernel_size=3, padding=0, stride=1), # 13
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[4],
                         outchannel=mb_backbone_channels[5], kernel_size=3, padding=0, stride=1), # 11
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[5],
                         outchannel=mb_backbone_channels[6], kernel_size=3, padding=1, stride=2), # 6
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[6],
                         outchannel=mb_backbone_channels[7], kernel_size=3, padding=0, stride=1),], # 4
        ),
        regression_head = dict(
            type='FCHead',
            conv_layer_number = 1,
            layers=[dict(type='Conv2dBlock',  inchannel=mb_backbone_channels[7],
                    outchannel=mb_backbone_channels[8], kernel_size=3, padding=1, stride=1)],
            fc_number = 3,
            fc_channels = [128, 128, face_landmark_num*2+rect_num*4],
        ),
        cls_head=dict(
            type='FCHead',
            conv_layer_number=1,
            layers=[dict(type='Conv2dBlock', inchannel=mb_backbone_channels[7],
                         outchannel=occ_head_channels, kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[128, 128, 2],
        ),
        occ_head=dict(
            type='FCHead',
            conv_layer_number=1,
            layers=[dict(type='Conv2dBlock', inchannel=mb_backbone_channels[7],
                         outchannel=occ_head_channels, kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[128, 128, face_landmark_num*2],
        ),
    ),

    # lip branch
    lip_branch=dict(
        type='Align282RoiBranch_v1',
        roi_start='image',
        feature_map_size=3,
        roi_aug=True,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio=0.05,
        translate_ratio=0.05,
        roialign=dict(type='RoiAlignLayer', pooled_h = 64, pooled_w=64,
                      sampling_ratio=1, spatial_scale=1),
        backbone=dict(
            type='StraightBackbone',
            conv_layer_number=7,
            layers=[dict(type='Conv2dBlock', inchannel=lb_backbone_channels[0],
                         outchannel=lb_backbone_channels[1], kernel_size=5, padding=0, stride=2), # 30
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[1],
                         outchannel=lb_backbone_channels[2], kernel_size=3, padding=0, stride=1), # 28
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[2],
                         outchannel=lb_backbone_channels[3], kernel_size=3, padding=1, stride=2), # 14
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[3],
                         outchannel=lb_backbone_channels[4], kernel_size=3, padding=0, stride=1), # 12
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[4],
                         outchannel=lb_backbone_channels[5], kernel_size=3, padding=0, stride=1), # 10
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[5],
                         outchannel=lb_backbone_channels[6], kernel_size=3, padding=1, stride=2), # 5
                    dict(type='Conv2dBlock', inchannel=lb_backbone_channels[6],
                         outchannel=lb_backbone_channels[7], kernel_size=3, padding=0, stride=1),], # 3
        ),
        regression_head=dict(
            type='FCHead',
            face_landmark_num = face_landmark_num,
            conv_layer_number = 1,
            layers=[dict(type='Conv2dBlock',  inchannel=lb_backbone_channels[7],
                    outchannel=lb_backbone_channels[8], kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[96, 96, mouth_landmark_num*2],  # 第二次融合将在此处进行
        ),
    ),

    # eye branch
    eye_branch=dict(
        type='Align282RoiBranch_v1',
        roi_start='image',
        feature_map_size=3,
        roi_aug=True,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio=0.05,
        translate_ratio=0.05,
        roialign=dict(type='RoiAlignLayer', pooled_h=48, pooled_w=48,
                      sampling_ratio=1, spatial_scale=1),
        backbone=dict(
            type='StraightBackbone',
            conv_layer_number=7,
            layers=[dict(type='Conv2dBlock', inchannel=eb_backbone_channels[0],
                         outchannel=eb_backbone_channels[1], kernel_size=5, padding=0, stride=2),  # 22
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[1],
                         outchannel=eb_backbone_channels[2], kernel_size=3, padding=0, stride=1),  # 20
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[2],
                         outchannel=eb_backbone_channels[3], kernel_size=3, padding=0, stride=1),  # 18
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[3],
                         outchannel=eb_backbone_channels[4], kernel_size=3, padding=1, stride=2),  # 9
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[4],
                         outchannel=eb_backbone_channels[5], kernel_size=3, padding=0, stride=1),  # 7
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[5],
                         outchannel=eb_backbone_channels[6], kernel_size=3, padding=0, stride=1),  # 5
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[6],
                         outchannel=eb_backbone_channels[7], kernel_size=3, padding=0, stride=1),], # 3
        ),
        regression_head=dict(
            type='FCHead',
            face_landmark_num=face_landmark_num,
            conv_layer_number=1,
            layers=[dict(type='Conv2dBlock', inchannel=eb_backbone_channels[7],
                         outchannel=eb_backbone_channels[8], kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[96, 96, eye_landmark_num * 2],  # 第二次融合将在此处进行
        ),
    ),

    # eyebrow branch
    eyebrow_branch=dict(
        type='Align282RoiBranch_v1',
        roi_start='feature',
        feature_map_size=2,
        roi_aug=False,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio=0.12,
        translate_ratio=0.15,
        roialign=dict(type='RoiAlignLayer', pooled_h=16, pooled_w=16,
                      sampling_ratio=1, spatial_scale=0.25),
        backbone=dict(
            type='StraightBackbone',
            conv_layer_number=5,
            layers=[dict(type='ResnetBlock', inchannel=ebb_backbone_channels[1],
                         outchannel=ebb_backbone_channels[2], kernel_size=3, padding=1, stride=2),  # 8
                    dict(type='ResnetBlock', inchannel=ebb_backbone_channels[2],
                         outchannel=ebb_backbone_channels[3], kernel_size=3, padding=1, stride=1),  # 8
                    dict(type='ResnetBlock', inchannel=ebb_backbone_channels[3],
                         outchannel=ebb_backbone_channels[4], kernel_size=3, padding=1, stride=2),  # 4
                    dict(type='ResnetBlock', inchannel=ebb_backbone_channels[4],
                         outchannel=ebb_backbone_channels[5], kernel_size=3, padding=1, stride=1),  # 4
                    dict(type='ResnetBlock', inchannel=ebb_backbone_channels[5],
                         outchannel=ebb_backbone_channels[6], kernel_size=3, padding=1, stride=2),],  # 2
        ),
        regression_head=dict(
            type='FCHead',
            face_landmark_num=face_landmark_num,
            conv_layer_number=1,
            layers=[dict(type='ResnetBlock', inchannel=ebb_backbone_channels[6],
                         outchannel=ebb_backbone_channels[7], kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[64, 64, eyebrow_landmark_num * 2],  # 第二次融合将在此处进行
        ),
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

'''
losses = dict(
    alignment = dict(
        type = 'face-alignment-loss',
        losses2 = [dict(type='smoothl1_with_threshold', name = 'rect_loss', weight=2),
                  dict(type='softwing_loss', thres1=1, thres2=10, curvature=0.3, name = 'face282_lip_loss', weight=128),
                  dict(type='smoothl1_with_threshold', name = 'face282lipdif_loss', weight=4),
                  dict(type='softwing_loss', thres1=1, thres2=10, curvature=0.3, name='face282_eyelid_loss', weight=96),
                  dict(type='smoothl1_with_threshold', name='face282_iris_loss', weight=1),
                  dict(type='softwing_loss', thres1=1, thres2=10, curvature=0.3, name='face282_eyebrow_loss', weight=52),
                  dict(type='smoothl1_with_threshold', name='face282_oripart58_loss', weight=1),
                  dict(type='smoothl1_with_threshold', name='face282_shift_loss', weight=1.5),
                  dict(type='smoothl1_with_threshold', name='rect_shift_loss', weight=1.5),
                  dict(type='smoothl1_with_threshold', name='face286_face106_loss', weight=0.5),
                  dict(type='smoothl1_with_threshold', name='contour_fitting_loss', weight=2.0)],
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
'''
