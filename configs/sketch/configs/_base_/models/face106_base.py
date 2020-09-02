model = dict(
    type = 'face-106-v1',
    crop_size = 112,
    landmark_num = 106,
    feature_map_size = 4,
    BN_type = 'BatchNormal',
    Relutype = 'PRelu',
    backbone = dict(
        type = 'tiny-single-backbone',
        conv_layer_number = 9,
        layer = [['conv2d', 8, 5, 0, 2],   # 54  # [block_name, channel, kernel_size, padding, stride]
                 ['avgpool', -1, 2, 0, 2], # 27
                 ['conv2d', 16, 3, 0, 1],  # 25
                 ['conv2d', 32, 3, 0, 1],  # 23
                 ['conv2d', 32, 3, 1, 2],  # 12
                 ['conv2d', 48, 3, 0, 1],  # 10
                 ['conv2d', 64, 3, 0, 1],  # 8
                 ['conv2d', 96, 3, 1, 2],  # 4
                 ['conv2d', 128, 3, 0, 1]],# 4
    ),
    regression_head = dict(
        type='fc-regression',
        conv_layer_number = 1,
        layer = [['conv2d', 128, 3, 1, 1]],
        fc_number = 3,
        fc_channels = [128, 128],
    ),
    cls_head=dict(
        type='fc-cls',
        conv_layer_number=1,
        layer=[['conv2d', 40, 3, 1, 1]],
        fc_number=3,
        fc_channels=[128, 128],
    ),
    occ_head=dict(
        type='occ-cls',
        conv_layer_number=1,
        layer=[['conv2d', 40, 3, 1, 1]],
        fc_number=3,
        fc_channels=[128, 128],
    ),
)

loss_alignment = dict(
    type = 'face_alignment_loss',
    losses = [dict(type ='softwing_loss', thres1 = 1, thres2 = 10, curvature=0.3, name = 'face_loss', weight=106),
              dict(type ='softwing_loss', thres1 = 1, thres2 = 10, curvature=0.3, name = 'contour_fitting_loss', weight=30),
              dict(type ='smoothl1_with_threshold', name = 'eyedif_left_loss', weight=10),
              dict(type ='smoothl1_with_threshold', name = 'eyedif_right_loss', weight=10),
              dict(type ='smoothl1_with_threshold', name = 'lipdif_loss', weight=5),
              dict(type ='softwing_loss', thres1 = 1, thres2 = 10, curvature=0.3, name = '3d_loss', weight=60),
              dict(type ='smoothl1_with_threshold', name='shift_loss', weight=1.5)]
)
loss_cls = dict(
    type = 'face_cls_loss',
    losses_with_weight = [dict(type = 'CrossEntropyLoss', name = 'cls_loss', weight=1)],
)

loss_occ = dict(
    type = 'face_occ_loss',
    losses_with_weight = [dict(type = 'CrossEntropyLoss', name = 'occ_loss', weight=1)],
)
