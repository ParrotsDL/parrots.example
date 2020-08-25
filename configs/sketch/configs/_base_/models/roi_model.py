model = dict(
    input_size = 256,
    landmark_num = 282,
    BN_type = 'BatchNormal',
    Relutype = 'PRelu',
    main_branch = dict(
        type = 'roi-main-v1',
        input_size = 128,
        landmark_num = 106,
        feature_map_size = 4,
        backbone_1=dict(
            type='tiny-single-backbone',
            conv_layer_number=1,
            layer=[['conv2d', 8, 5, 2, 2]]  # 64  # [block_name, channel, kernel_size, padding, stride]
        ),
        backbone2=dict(
            type='tiny-single-backbone',
            conv_layer_number=7,
            layer=[['avgpool', -1, 2, 0, 2],  # 32
                   ['conv2d', 16, 3, 0, 1],  # 30
                   ['conv2d', 24, 3, 1, 2],  # 15
                   ['conv2d', 32, 3, 0, 1],  # 13
                   ['conv2d', 32, 3, 0, 1],  # 11
                   ['conv2d', 48, 3, 1, 2],  # 6
                   ['conv2d', 64, 3, 1, 1]]  # 4
        ),
        regression_head=dict(
            type='fc-regression',
            conv_layer_number=1,
            layer=[['conv2d', 80, 3, 1, 1]],
            fc_number=3,
            fc_channels=[128, 128],   # 这里的最后一层全连接其实是多个全连接，包括回归主分支的106*2个坐标和回归各个部件框
        ),
        cls_head=dict(
            type='fc-cls',
            conv_layer_number=1,
            layer=[['conv2d', 40, 3, 1, 1]],
            fc_number=3,
            fc_channels=[128, 128],
        ),
        occ_head=dict(
            type='fc-occ',
            conv_layer_number=1,
            layer=[['conv2d', 40, 3, 1, 1]],
            fc_number=3,
            fc_channels=[128, 128],
        ),
    ),
    mouth_branch=dict(
        roi_start = 'image',
        spatial_scale = 1,
        roi_size = 64,
        landmark_num = 84,
        feature_map_size = 3,
        roi_aug = True,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio = 0.05,
        translate_ratio = 0.05,
        backbone2=dict(
            type='tiny-single-backbone',
            conv_layer_number=8,
            layer=[['conv2d', 8, 5, 2, 0],   # 30
                   ['conv2d', 16, 3, 0, 1],  # 28
                   ['conv2d', 24, 3, 1, 2],  # 14
                   ['conv2d', 32, 3, 0, 1],  # 12
                   ['conv2d', 32, 3, 0, 1],  # 10
                   ['conv2d', 48, 3, 1, 2],  # 5
                   ['conv2d', 64, 3, 1, 0],  # 3
                   ['conv2d', 64, 3, 1, 1]]  # 3
        ),
        regression_head=dict(
            type='fc-regression',
            fc_number=3,
            fc_channels=[96, 96],  # 第二次融合将在此处进行
        )
    ),
    eye_branch=dict(
        roi_start='image',
        spatial_scale=1,
        roi_size=46,
        landmark_num=53,
        feature_map_size=4,
        roi_aug=True,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio=0.08,
        translate_ratio=0.08,
        backbone2=dict(
            type='tiny-single-backbone',
            conv_layer_number=8,
            layer=[['conv2d', 8, 5, 2, 0],  # 22
                   ['conv2d', 16, 3, 0, 1],  # 20
                   ['conv2d', 24, 3, 1, 2],  # 10
                   ['conv2d', 32, 3, 0, 1],  # 8
                   ['conv2d', 32, 3, 0, 1],  # 6
                   ['conv2d', 48, 3, 0, 1],  # 4
                   ['conv2d', 64, 3, 1, 1],  # 4
                   ['conv2d', 64, 3, 1, 1]]  # 4
        ),
        regression_head=dict(
            type='fc-regression',
            fc_number=3,
            fc_channels=[96, 96],  # 第二次融合将在此处进行
        )
    ),
    eyebrow_branch=dict(
        roi_start='feature',
        spatial_scale=4,
        roi_size=16,
        landmark_num=22,
        feature_map_size=4,
        roi_aug=False,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio=0.08,
        translate_ratio=0.08,
        backbone2=dict(
            type='tiny-single-backbone',
            conv_layer_number=6,
            layer=[['conv2d', 8, 3, 0, 1],   # 14
                   ['conv2d', 8, 3, 0, 1],  # 12
                   ['conv2d', 16, 3, 1, 2],  # 6
                   ['conv2d', 16, 3, 0, 1],  # 4
                   ['conv2d', 24, 3, 0, 1],  # 2
                   ['conv2d', 32, 3, 0, 1]]  # 2
        ),
        regression_head=dict(
            type='fc-regression',
            fc_number=3,
            fc_channels=[64, 64],  # 第二次融合将在此处进行
        ),
    ),
)

loss_alignment = dict(
    type = 'face_alignment_loss',
    losses = [dict(type='smoothl1_with_threshold', name = 'rect_loss', weight=2),
              dict(type='softwing_loss', thres1=1, thres2=10, curvature=0.3, name = 'face282_lip_loss', weight=128),
              dict(type='smoothl1_with_threshold', name = 'lipdif_loss', weight=4),
              dict(type='softwing_loss', thres1=1, thres2=10, curvature=0.3, name='face282_eyelid_loss', weight=96),
              dict(type='smoothl1_with_threshold', name='face282_iris_loss', weight=1),
              dict(type='softwing_loss', thres1=1, thres2=10, curvature=0.3, name='face282_eyebrow_loss', weight=52),
              dict(type='smoothl1_with_threshold', name='face282_oripart58_loss', weight=1),
              dict(type='smoothl1_with_threshold', name='face282_shift_loss', weight=1.5),
              dict(type='smoothl1_with_threshold', name='rect_shift_loss', weight=1.5),
              dict(type='smoothl1_with_threshold', name='face286_face106_loss', weight=0.5),
              dict(type='smoothl1_with_threshold', name='contour_fitting_loss', weight=2.0)],
)
loss_cls = dict(
    type = 'face_cls_loss',
    losses_with_weight = [dict(type = 'CrossEntropyLoss', name = 'cls_loss', weight=1)],
)

loss_occ = dict(
    type = 'face_occ_loss',
    losses_with_weight = [dict(type = 'CrossEntropyLoss', name = 'occ_loss', weight=1)],
)
