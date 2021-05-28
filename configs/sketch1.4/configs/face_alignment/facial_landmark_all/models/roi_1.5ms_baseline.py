#variables
mb_backbone_channels = [1, 8, 16, 16, 32, 40, 48, 64, 80]#[1, 6, 8, 16, 32, 40, 48, 64, 80, 80]
lb_backbone_channels = [1, 6, 16, 16, 24, 32, 32, 64, 80]
eb_backbone_channels = [1, 6, 16, 16, 24, 24, 32, 48, 48]
ebb_backbone_channels = [1, 4, 8, 8, 8, 12, 12, 16]
cls_head_channels = 64
occ_head_channels = 40
eyestate_head_channels = 40
face_landmark_num = 106
rect_num = 5
mouth_landmark_num_main = 20
mouth_landmark_num = 64+20
eye_landmark_num_main = 10
eye_landmark_num = 24+19+10
eyebrow_landmark_num_main = 9
eyebrow_landmark_num = 13+9
model_crop_size = 192
# just for test on 106, need remove
backbone_channels = mb_backbone_channels
model_landmark_num = 282

#module
model = dict(
    type = 'Align282_Roi_v4',
    crop_size = model_crop_size,
    rect_num = rect_num,
    landmark_num = face_landmark_num+mouth_landmark_num+eye_landmark_num*2+eyebrow_landmark_num*2,
    BN_type = 'BN',
    Relu_type = 'PReLU',
    init_relu = 'leaky_relu',
    init_type = 'kaiming_uniform',
    freeze_main = False,
    cls_path = 'mainbranch_cls.pth',
    occ_path = 'mainbranch_occ.pth',
    # main 106
    main_branch=dict(
        type='Align282MainBranch_v2',
        feature_map_size = 4,
        backbone1=dict(
            type='StraightBackbone',
            conv_layer_number=0,
            #layers=[dict(type='InterpolateLayer', output_size=[112, 112])]
            layers=[]
        ),
        backbone=dict(
            type='StraightBackbone',
            conv_layer_number=9,
            layers=[dict(type='InterpolateLayer',output_size=[112,112]),
                    dict(type='Conv2dBlock',   inchannel=mb_backbone_channels[0],
                        outchannel=mb_backbone_channels[1], kernel_size=5, padding=0, stride=2),
                    dict(type='AvgPool', kernel_size=2, stride=2, padding=0),
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[1],
                        outchannel=mb_backbone_channels[2], kernel_size=3, padding=0, stride=1),
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[2],
                        outchannel=mb_backbone_channels[3], kernel_size=3, padding=0, stride=1),
                    dict(type='Conv2dBlock',   inchannel=mb_backbone_channels[3],
                        outchannel=mb_backbone_channels[4], kernel_size=3, padding=1, stride=2),
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[4],
                        outchannel=mb_backbone_channels[5], kernel_size=3, padding=0, stride=1),
                    dict(type='Conv2dBlock', inchannel=mb_backbone_channels[5],
                        outchannel=mb_backbone_channels[6], kernel_size=3, padding=0, stride=1),
                    dict(type='Conv2dBlock',   inchannel=mb_backbone_channels[6],
                        outchannel=mb_backbone_channels[7], kernel_size=3, padding=1, stride=2)]
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
            layers=[dict(type='Conv2dBlock',  inchannel=mb_backbone_channels[7],
                        outchannel=cls_head_channels, kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[64, 64, 1],
        ),
        occ_head=dict(
            type='FCHead',
            conv_layer_number=1,
            layers=[dict(type='Conv2dBlock', inchannel=mb_backbone_channels[7],
                        outchannel=occ_head_channels, kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[64, 64, face_landmark_num*2],
        ),
    ),

    # lip branch
    lip_branch=dict(
        type='Align282RoiBranch_v5',
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
            face_landmark_num = mouth_landmark_num_main,
            conv_layer_number = 1,
            layers=[dict(type='Conv2dBlock',  inchannel=lb_backbone_channels[7],
                        outchannel=lb_backbone_channels[8], kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[96, 96, mouth_landmark_num*2],  # 第二次融合将在此处进行
        ),
    ),

    # eye branch
    eye_branch=dict(
        type='Align282RoiBranch_with_eyestate_maintransform',
        roi_start='image',
        feature_map_size=4,
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
                         outchannel=eb_backbone_channels[3], kernel_size=3, padding=1, stride=2),  # 10
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[3],
                         outchannel=eb_backbone_channels[4], kernel_size=3, padding=0, stride=1),  # 8
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[4],
                         outchannel=eb_backbone_channels[5], kernel_size=3, padding=0, stride=1),  # 6
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[5],
                         outchannel=eb_backbone_channels[6], kernel_size=3, padding=0, stride=1),  # 4
                    dict(type='Conv2dBlock', inchannel=eb_backbone_channels[6],
                         outchannel=eb_backbone_channels[7], kernel_size=3, padding=1, stride=1),], # 4
        ),
        regression_head=dict(
            type='FCHead',
            face_landmark_num=eye_landmark_num_main,
            conv_layer_number=1,
            layers=[dict(type='Conv2dBlock', inchannel=eb_backbone_channels[7],
                         outchannel=eb_backbone_channels[8], kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[96, 96, eye_landmark_num * 2],  # 第二次融合将在此处进行
        ),
        eyestate_head=dict(
            type='FCHead',
            conv_layer_number=1,
            layers=[dict(type='Conv2dBlock', inchannel=eb_backbone_channels[7],
                         outchannel=eyestate_head_channels, kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[64, 64, 2],  # 第二次融合将在此处进行
        ),
    ),

    # eyebrow branch
    eyebrow_branch=dict(
        type='Align282RoiBranch_v5',
        roi_start='image',
        feature_map_size=2,
        roi_aug=False,  # 当设置为true时，将使用预测框的augment进行训练，为False时，使用GT框的augment进行训练
        scale_ratio=0.12,
        translate_ratio=0.15,
        roialign=dict(type='RoiAlignLayer', pooled_h=36, pooled_w=36,
                      sampling_ratio=1, spatial_scale=1.0),
        backbone=dict(
            type='StraightBackbone',
            conv_layer_number=6,
            layers=[dict(type='Conv2dBlock', inchannel=ebb_backbone_channels[0],
                         outchannel=ebb_backbone_channels[1], kernel_size=5, padding=0, stride=2),  # 16
                    dict(type='Conv2dBlock', inchannel=ebb_backbone_channels[1],
                         outchannel=ebb_backbone_channels[2], kernel_size=3, padding=0, stride=1),  # 14
                    dict(type='Conv2dBlock', inchannel=ebb_backbone_channels[2],
                         outchannel=ebb_backbone_channels[3], kernel_size=3, padding=0, stride=1),  # 12
                    dict(type='Conv2dBlock', inchannel=ebb_backbone_channels[3],
                         outchannel=ebb_backbone_channels[4], kernel_size=3, padding=1, stride=2),  # 6
                    dict(type='Conv2dBlock', inchannel=ebb_backbone_channels[4],
                         outchannel=ebb_backbone_channels[5], kernel_size=3, padding=0, stride=1),  # 4
                    dict(type='Conv2dBlock', inchannel=ebb_backbone_channels[5],
                         outchannel=ebb_backbone_channels[6], kernel_size=3, padding=0, stride=1)],  # 2
        ),
        regression_head=dict(
            type='FCHead',
            face_landmark_num=eyebrow_landmark_num_main,
            conv_layer_number=1,
            layers=[dict(type='Conv2dBlock', inchannel=ebb_backbone_channels[6],
                         outchannel=ebb_backbone_channels[7], kernel_size=3, padding=1, stride=1)],
            fc_number=3,
            fc_channels=[32, 32, eyebrow_landmark_num * 2],  # 第二次融合将在此处进行
        ),
    ),
)
# loss
losses = dict(
    losses_info = [
        dict(type = 'LandmarkLoss', 
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'main',
             part = ['face_106pt_all'],
             weight = 106),
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "main",
             part = ['face_106pt_all'],
             weight = 1.2),
        dict(type = "ContourFittingLoss",
             base_loss=dict(type="CF_SoftWingLoss"),
             weight = 60),
        dict(type = "RectLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             rects = ['mouth','eye','eyebrow'],
             weight = 1.5,
             loss_name="all_rect"),
        dict(type = "RectShiftLoss",
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             rects = ['mouth','eye','eyebrow'],
             weight=1.2,
             loss_name="all_rect_shift"), # until here, loss of main branch
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'mouth',
             part = ['lip_64pt'],
             weight = 128),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'mouth',
             part = ['lip_20pt'],
             weight = 1),
        dict(type = 'DiffLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'mouth',
             part = ['lip_64pt'],
             close_weight=20.,use_close_weight=True,
             weight = 5),
        dict(type = 'SpecialPointLoss',
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'mouth',
             part = ['lip_64pt'],
             special_feature="Static3D",
             weight = 30),
        dict(type = 'SpecialPointLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'mouth',
             part = ['lip_64pt'],
             special_feature="Inner",
             weight = 5),
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "mouth",
             part = ['lip_20pt', 'lip_64pt'],
             weight = 1.2,
             loss_name="all_mouth_shift"), # until here, loss of mouth branch
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'eye',
             part = ['eyelid_24pt'],
             weight = 96),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'eye',
             part = ['iris_19pt'],
             weight = 38*1.5),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'eye',
             part = ['eyelid_10pt'],
             weight = 1), 
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "eye",
             part = ['eyelid_24pt', 'iris_19pt', 'eyelid_10pt'],
             weight = 1.2,
             loss_name="all_eye_shift"), # until here, loss of eye branch
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SoftWingLoss", thres1=0.6),
             branch = 'eyebrow',
             part = ['eyebrow_13pt'],
             weight = 26),
        dict(type = 'LandmarkLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = 'eyebrow',
             part = ['eyebrow_9pt'],
             weight = 1),
        dict(type = 'ShiftLoss',
             base_loss=dict(type="SmoothL1LossWithThreshold"),
             branch = "eyebrow",
             part = ['eyebrow_13pt','eyebrow_9pt'],
             weight = 1.2,
             loss_name="all_eyebrow_shift"), # until here, loss of left eyebrow branch
    ]
)

metrics = dict(
    metrics_info = [
        dict(type = "RectMetric",
             base_metric=dict(type="BaseNMEMetric"),
             rects = ['mouth','eye','eyebrow'],
             metric_name="all_rect"),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["main"],
            part=["face_106pt_all"],
            metric_name="face_106"
        ),
        dict(type="ContourFittingMetric",
            base_metric=dict(type="ContourNMEMetric")),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["mouth"],
            part=["lip_64pt"],
            metric_name="lip_64"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["eye"],
            part=["eyelid_24pt"],
            metric_name="eyelid_24"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["eyebrow"],
            part=["eyebrow_13pt"],
            metric_name="eyebrow_13"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["eye"],
            part=["iris_19pt"],
            metric_name="iris_19"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["mouth","eye","eye","eyebrow"],
            part=["lip_64pt","eyelid_24pt","iris_19pt","eyebrow_13pt"],
            metric_name="roi_part"
        ),
        dict(type="LandmarkMetric",
            base_metric=dict(type="BaseNMEMetric"),
            branch=["main", "mouth","eye","eye","eyebrow"],
            part=["face_106pt_all", "lip_64pt","eyelid_24pt","iris_19pt","eyebrow_13pt"],
            metric_name="face_282"
        ),
    ]
)


