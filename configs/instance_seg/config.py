# config.py
import os.path
import copy

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))


# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


class Config(object):
    """
    Holds the config for various networks.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


# Datasets
dataset_base = Config({
    'name': 'Base Dataset',

    'train_images': './data/coco/images/',
    'train_info':   'path_to_annotation_file',

    'valid_images': './data/coco/images/',
    'valid_info':   'path_to_annotation_file',

    'has_gt': True,
    'class_names': COCO_CLASSES,
    'label_map': None
})

MHP_dataset = dataset_base.copy({
    'name': 'MHP',
    'train_images': '/mnt/lustre/DATAshare/coco2014/train2014',
    'train_info':   'path_to_annotation_file',
    'valid_images': '/data/instance_seg/resize_640/LV-MHP-v2-val/image',
    'valid_info':   '/data/instance_seg/resize_640/LV-MHP-v2-val/LV-MHP-v2-val.json',
})

MHP_dataset_1 = dataset_base.copy({
    'name': 'MHP',
    'train_images': '/mnt/lustre/depthgroup/datasets/instance_seg/LV-MHP-v2-val/image',
    'train_info':   '/mnt/lustre/depthgroup/datasets/instance_seg/LV-MHP-v2-val/LV-MHP-v2-val.json',

    'valid_images': '/data/instance_seg/resize_640/LV-MHP-v2-val/image',
    'valid_info':   '/data/instance_seg/resize_640/LV-MHP-v2-val/LV-MHP-v2-val.json',
})

coco2014_dataset = dataset_base.copy({
    'name': 'COCO 2014',
    'train_images': '/mnt/lustre/DATAshare/coco2014/train2014',
    'train_info': '/mnt/lustre/yanjiawei/Data_t1/instance_seg/instances_train2014_person.json',
    #'train_info': '/mnt/lustre/yanjiawei/Data_t1/instance_seg/instances_train2014.json',
    'valid_images': '/mnt/lustre/DATAshare/coco2014/val2014',
    'valid_info': '/mnt/lustre/yanjiawei/Data_t1/instance_seg/instances_val2014.json',
    'label_map': COCO_LABEL_MAP
})

coco2017_dataset = dataset_base.copy({
    'name': 'COCO 2017',
    'train_images': '/mnt/lustre/share/DSK/datasets/mscoco2017/train2017',
    'train_info': '/mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_train2017.json',
    'valid_images': '/mnt/lustre/share/DSK/datasets/mscoco2017/test2017',
    'valid_info': '/mnt/lustre/share/DSK/datasets/mscoco2017/instances_val2017.json',
    'label_map': COCO_LABEL_MAP
})

coco2017_testdev_dataset = dataset_base.copy({
    'name': 'COCO 2017 Test-Dev',

    'valid_images': '/mnt/lustre/share/DSK/datasets/mscoco2017/test2017',
    'valid_info': '/mnt/lustre/share/DSK/datasets/mscoco2017/instances_val2017.json',
    'label_map': COCO_LABEL_MAP
})

# Backbones
from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone, BasicBlock, ShuffleNetV2, MobileNetV2, EfficientNet
from torchvision.models.vgg import cfg as vggcfg
from math import sqrt
import torch


resnet_transform = Config({
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
})

vgg_transform = Config({
    # Note that though vgg is traditionally BRG,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
})

darknet_transform = Config({
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
})

resnet101_backbone = Config({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],[64, 128, 256, 512],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
    'use_pixel_scales': False,
})

resnet101_gn_backbone = Config({
    'name': 'ResNet101_GN',
    'path': 'R-101-GN.pkl',
    'type': ResNetBackboneGN,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
    'use_pixel_scales': False,
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],[64, 128, 256, 512],),
    'transform': resnet_transform,
})

resnet18_backbone = resnet101_backbone.copy({
    'name': 'ResNet18',
    'path': 'resnet18-5c106cde.pth',
    'type': ResNetBackbone,
    'args': ([2, 2, 2, 2], [64, 128, 256, 512], True, BasicBlock,),
    'transform': resnet_transform,
})


shufflenetv2_backbone = Config({
    'name': 'ShuffleNetV2',
    'path': 'shufflenetv2_x1-5666bf0f80.pt',
    'type': ShuffleNetV2,
    'args': ([4, 8, 4], [24, 116, 232, 464, 1024], 2),
    'transform': resnet_transform,
})

shufflenetv2_x0_5_backbone = Config({
    'name': 'ShuffleNetV2_x0_5',
    'path': 'shufflenetv2_x0.5-f707e7126e.pth',
    'type': ShuffleNetV2,
    'args': ([4, 8, 4], [24, 48, 96, 192, 1024], 2),
    'transform': resnet_transform,
})

mobilenetv2_backbone = Config({
    'name': 'MobileNetV2',
    'path': 'mobilenet_v2-b0353104.pth',
    'type': MobileNetV2,
    'args': ([
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ],),
    'transform': resnet_transform,
    'selected_layers': [2, 4, 7],        
    'pred_scales': [[24], [48], [96], [192], [384]],
    'pred_aspect_ratios': [ [[1, 1/sqrt(2), sqrt(2)]] ]*6,
    'use_pixel_scales': True,
})


efficientnet_backbone = Config({
    'name': 'EfficientNet',
    'path': 'efficientnet-b0-08094119.pth',
    'type': EfficientNet,
    'args': (),
    'transform': resnet_transform,
})

darknet53_backbone = Config({
    'name': 'DarkNet53',
    'path': 'darknet53.pth',
    'type': DarkNetBackbone,
    'args': ([1, 2, 8, 8, 4],),
    'transform': darknet_transform,

    'selected_layers': list(range(3, 9)),
    'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
    'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    'use_pixel_scales': False,
})

vgg16_arch = [[64, 64],
              [ 'M', 128, 128],
              [ 'M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              [ 'M', 512, 512, 512],
              [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = Config({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),
    'transform': vgg_transform,

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]]*6,
    'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
    'use_pixel_scales': False,
})

mask_type = Config({
    # Direct produces masks directly as the output of each pred module.
    # Parameters: mask_size, use_gt_bboxes
    'direct': 0,

    # Lincomb produces coefficients as the output of each pred module then uses those coefficients
    # to linearly combine features from an earlier convout to create image-sized masks.
    # Parameters:
    #   - masks_to_train (int): Since we're producing (near) full image masks, it'd take too much
    #                           vram to backprop on every single mask. Thus we select only a subset.
    #   - mask_proto_src (int): The input layer to the mask prototype generation network. This is an
    #                           index in backbone.layers. Use to use the image itself instead.
    #   - mask_proto_net (list<tuple>): A list of layers in the mask proto network with the last one
    #                                   being where the masks are taken from. Each conv layer is in
    #                                   the form (num_features, kernel_size, **kwdargs). An empty
    #                                   list means to use the source for prototype masks. If the
    #                                   kernel_size is negative, this creates a deconv layer instead.
    #                                   If the kernel_size is negative and the num_features is None,
    #                                   this creates a simple bilinear interpolation layer instead.
    #   - mask_proto_bias (bool): Whether to include an extra coefficient that corresponds to a proto
    #                             mask of all ones.
    #   - mask_proto_prototype_activation (func): The activation to apply to each prototype mask.
    #   - mask_proto_mask_activation (func): After summing the prototype masks with the predicted
    #                                        coeffs, what activation to apply to the final mask.
    #   - mask_proto_coeff_activation (func): The activation to apply to the mask coefficients.
    #   - mask_proto_crop (bool): If True, crop the mask with the predicted bbox during training.
    #   - mask_proto_crop_expand (float): If cropping, the percent to expand the cropping bbox by
    #                                     in each direction. This is to make the model less reliant
    #                                     on perfect bbox predictions.
    #   - mask_proto_loss (str [l1|disj]): If not None, apply an l1 or disjunctive regularization
    #                                      loss directly to the prototype masks.
    #   - mask_proto_binarize_downsampled_gt (bool): Binarize GT after dowsnampling during training?
    #   - mask_proto_normalize_mask_loss_by_sqrt_area (bool): Whether to normalize mask loss by sqrt(sum(gt))
    #   - mask_proto_reweight_mask_loss (bool): Reweight mask loss such that background is divided by
    #                                           #background and foreground is divided by #foreground.
    #   - mask_proto_grid_file (str): The path to the grid file to use with the next option.
    #                                 This should be a numpy.dump file with shape [numgrids, h, w]
    #                                 where h and w are w.r.t. the mask_proto_src convout.
    #   - mask_proto_use_grid (bool): Whether to add extra grid features to the proto_net input.
    #   - mask_proto_coeff_gate (bool): Add an extra set of sigmoided coefficients that is multiplied
    #                                   into the predicted coefficients in order to "gate" them.
    #   - mask_proto_prototypes_as_features (bool): For each prediction module, downsample the prototypes
    #                                 to the convout size of that module and supply the prototypes as input
    #                                 in addition to the already supplied backbone features.
    #   - mask_proto_prototypes_as_features_no_grad (bool): If the above is set, don't backprop gradients to
    #                                 to the prototypes from the network head.
    #   - mask_proto_remove_empty_masks (bool): Remove masks that are downsampled to 0 during loss calculations.
    #   - mask_proto_reweight_coeff (float): The coefficient to multiple the forground pixels with if reweighting.
    #   - mask_proto_coeff_diversity_loss (bool): Apply coefficient diversity loss on the coefficients so that the same
    #                                             instance has similar coefficients.
    #   - mask_proto_coeff_diversity_alpha (float): The weight to use for the coefficient diversity loss.
    #   - mask_proto_normalize_emulate_roi_pooling (bool): Normalize the mask loss to emulate roi pooling's affect on loss.
    #   - mask_proto_double_loss (bool): Whether to use the old loss in addition to any special new losses.
    #   - mask_proto_double_loss_alpha (float): The alpha to weight the above loss.
    'lincomb': 1,
})

# Self explanitory. For use with mask_proto_*_activation
activation_func = Config({
    'tanh':    torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu':    lambda x: torch.nn.functional.relu(x, inplace=True),
    'none':    lambda x: x,
})


fpn_base = Config({
    # The number of features to have in each FPN layer
    'num_features': 256,

    # The upsampling mode used
    'interpolation_mode': 'bilinear',

    # The number of extra layers to be produced by downsampling starting at P5
    'num_downsample': 1,

    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': False,

    # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
    # This is just here for backwards compatibility
    'pad': True,
})

# Configs
coco_base_config = Config({
    'dataset': coco2014_dataset,
    'num_classes': 81, # This should include the background class

    'max_iter': 400000,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 360000, 400000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'bbox_alpha': 1.5,
    'mask_alpha': 6.125, # Some funky equation. Don't worry about it.

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # See mask_type for details.
    'mask_type': mask_type.direct,
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    'mask_proto_bias': False,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
    'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': True,
    'mask_proto_crop_random_expand': False,
    'mask_proto_crop_expand': 0,
    'mask_proto_loss': None,
    'mask_proto_binarize_downsampled_gt': True,
    'mask_proto_normalize_mask_loss_by_sqrt_area': False,
    'mask_proto_reweight_mask_loss': False,
    'mask_proto_grid_file': 'data/grid.npy',
    'mask_proto_use_grid':  False,
    'mask_proto_coeff_gate': False,
    'mask_proto_prototypes_as_features': False,
    'mask_proto_prototypes_as_features_no_grad': False,
    'mask_proto_remove_empty_masks': False,
    'mask_proto_reweight_coeff': 1,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_coeff_diversity_alpha': 1,
    'mask_proto_normalize_emulate_roi_pooling': False,
    'mask_proto_double_loss': False,
    'mask_proto_double_loss_alpha': 1,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': True,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # rotate image
    'augment_rotate': False,
    # maximum angle for image rotation
    'rotate_max_angle': 0,
    # crop less than half bounding box
    'crop_half_person': False,
    'affine_transform': 0.1,
    'affine_scale': 0.05,
    'affine_rotate': 0,
    'input_channel': 3,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': None,

    # Use the same weights for each network head
    'share_prediction_module': False,

    # For hard negative mining, instead of using the negatives that are leastl confidently background,
    # use negatives that are most confidently not background.
    'ohem_use_most_confident': False,

    # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
    'use_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,
    
    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'use_sigmoid_focal_loss': False,

    # Use class[0] to be the objectness score and class[1:] to be the softmax predicted class.
    # Note: at the moment this is only implemented if use_focal_loss is on.
    'use_objectness_score': False,

    # Adds a global pool + fc layer to the smallest selected layer that predicts the existence of each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_class_existence_loss': False,
    'class_existence_alpha': 1,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.
    'extra_head_net': None,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {'kernel_size': 3, 'padding': 1},

    # Add extra layers between the backbone and the network heads
    # The order is (bbox, conf, mask)
    'extra_layers': (0, 0, 0),

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.5,

    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 1,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size. If preserve_aspect_ratio is False, min_size is ignored.
    'min_size': 200,
    'max_size': 300,
    
    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to have a separate branch whose sole purpose is to act as the coefficients for coeff_diversity_loss
    # Remember to turn on coeff_diversity_loss, or these extra coefficients won't do anything!
    # To see their effect, also remember to turn on use_coeff_nms.
    'use_instance_coeff': False,
    'num_instance_coeffs': 64,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, uses the faster r-cnn resizing scheme.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the prediction module (c) from DSSD
    'use_prediction_module': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,
    
    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,

    'backbone': None,
    'name': 'base_config',

    # concatinate shallower layer as prtonet input
    'concat_shallow_layer': False,
    'shallow_layer_number': 1, #which output layer of backbone to concat for protonet input, for resnet this is 0, for mobilenet this is 1
    'proto_upsample_size': 121, # upsample protonet input to this size to concatinate with shallower layer
    'proto_in_channel': 152, # number of input channels after concat 
    'deconv_for_upsample': False, # use deconv for upsample instead of interpolation
})

yolact_base_config = coco_base_config.copy({
    'name': 'yolact_base',

    # Dataset stuff
    'dataset': coco2017_dataset,
    'num_classes': 81,

    # Image Size
    'max_size': 550,
    
    # Training params
    'lr_steps': (280000, 600000, 700000, 750000),
    'max_iter': 800000,
    
    # Backbone Settings
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,

        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[24], [48], [96], [192], [384]],
    }),

    # FPN Settings
    'fpn': fpn_base.copy({
        'use_conv_downsample': True,
        'num_downsample': 2,
    }),

    # Mask Settings
    'mask_type': mask_type.lincomb,
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
    'mask_proto_normalize_emulate_roi_pooling': True,

    # Other stuff
    'share_prediction_module': True,
    'extra_head_net': [(256, 3, {'padding': 1})],

    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,

    'crowd_iou_threshold': 0.7,

    'use_semantic_segmentation_loss': True,
})

yolact_im400_config = yolact_base_config.copy({
    'name': 'yolact_im400',

    'max_size': 400,
    'backbone': yolact_base_config.backbone.copy({
        'pred_scales': [[int(x[0] / yolact_base_config.max_size * 400)] for x in yolact_base_config.backbone.pred_scales],
    }),
})

yolact_im700_config = yolact_base_config.copy({
    'name': 'yolact_im700',

    'masks_to_train': 300,
    'max_size': 700,
    'backbone': yolact_base_config.backbone.copy({
        'pred_scales': [[int(x[0] / yolact_base_config.max_size * 700)] for x in yolact_base_config.backbone.pred_scales],
    }),
})

yolact_darknet53_config = yolact_base_config.copy({
    'name': 'yolact_darknet53',

    'backbone': darknet53_backbone.copy({
        'selected_layers': list(range(2, 5)),
        
        'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
    }),
})

yolact_resnet50_config = yolact_base_config.copy({
    'name': 'yolact_resnet50',

    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        
        'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
    }),
})

yolact_resnet18_config = yolact_base_config.copy({
    'name': 'yolact_resnet18',
    'num_classes': 2,
    'backbone': resnet18_backbone.copy({
        'selected_layers': list(range(1, 4)),       
        'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
    }),
    'lr_steps': (630000, 1350000, 1600000, 1700000),
    'max_iter': 2200000,   
})

yolact_resnet50_instance_config = yolact_resnet50_config.copy({
    'name': 'yolact_resnet50_instance',
    'num_classes': 2,
    'lr_steps': (800000, 1800000, 2000000, 2200000),
    'max_iter': 2500000, 
    'max_size': 481,
})

yolact_shufflenetv2_config = yolact_base_config.copy({
    'name': 'yolact_shufflenetv2',
    'num_classes': 2,
    'backbone': shufflenetv2_backbone.copy({
        'selected_layers': list(range(0, 3)),        
        'pred_scales': [[24], [48], [96], [192], [384]],
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
    })
})


yolact_mobilenetv2_config = yolact_base_config.copy({
    'name': 'yolact_mobilenetv2',
    'num_classes': 2,
    'backbone': mobilenetv2_backbone.copy(),
    'lr_steps': (630000, 1350000, 1600000, 1700000),
    'max_iter': 2200000,
})

yolact_mobilenetv2_481_config = yolact_base_config.copy({
    'name': 'yolact_mobilenetv2_481',
    'num_classes': 2,
    'backbone': mobilenetv2_backbone.copy(),
    'lr_steps': (630000, 1350000, 1600000, 1700000),
    'max_iter': 2500000,
    'max_size': 481,
})

yolact_mobilenetv2_compress21_config = yolact_mobilenetv2_config.copy({
    'name': 'yolact_mobilenetv2_compress21',
    'num_classes': 2,
    'mask_proto_net': [(128, 3, {'padding': 1})] * 3 + [(None, -121, {}), (128, 3, {'padding': 1})] + [(16, 1, {})],
    'fpn': yolact_base_config.fpn.copy({
        'num_features': 128,
    }),
    'max_size': 481,
    'lr_steps': (800000, 1800000, 2000000, 2200000),
    'max_iter': 2500000,    
})

yolact_mobilenetv2_concat_config = yolact_mobilenetv2_config.copy({
    'name': 'yolact_mobilenetv2_concat',
    'num_classes': 2,
    'mask_proto_net': [(128, 3, {'padding': 1})] * 3 + [(None, -241, {}), (128, 3, {'padding': 1})] + [(16, 1, {})],
    'fpn': yolact_base_config.fpn.copy({
        'num_features': 128,
    }),
    'max_size': 481,   
    'concat_shallow_layer': True,
    'proto_upsample_size': 121,
    'proto_in_channel': 152,
    'lr_steps': (800000, 1800000, 2000000, 2200000),
    'max_iter': 2500000, 
})

yolact_mobilenetv2_concat_32_config = yolact_mobilenetv2_config.copy({
    'name': 'yolact_mobilenetv2_concat_32',
    'num_classes': 2,
    'mask_proto_net': [(128, 3, {'padding': 1})] * 3 + [(None, -241, {}), (128, 3, {'padding': 1})] + [(32, 1, {})],
    'fpn': yolact_base_config.fpn.copy({
        'num_features': 128,
    }),
    'max_size': 481,   
    'concat_shallow_layer': True,
    'proto_upsample_size': 121,
    'proto_in_channel': 152,
    'lr_steps': (800000, 1800000, 2000000, 2200000),
    'max_iter': 2500000, 
})

yolact_mobilenetv2_concat_32_coco_config = yolact_mobilenetv2_config.copy({
    'name': 'yolact_mobilenetv2_concat_32_coco',
    'num_classes': 81,
    'lr_steps': (280000, 600000, 700000, 750000),
    'max_iter': 800000,
})

yolact_mobilenetv2_concat_32_641_config = yolact_mobilenetv2_concat_32_config.copy({
    'name': 'yolact_mobilenetv2_concat_32_641',
    'max_size': 641,   
    'proto_upsample_size': 161,
})

yolact_mobilenetv2_concat_32_deconv_config = yolact_mobilenetv2_concat_32_config.copy({
    'name': 'yolact_mobilenetv2_concat_32_deconv',
    'num_classes': 2,
    'mask_proto_net': [(128, 3, {'padding': 1})] * 3 + [(128, -3, {'padding': 1, 'stride':2}), (128, 3, {'padding': 1})] + [(32, 1, {})],
    'concat_shallow_layer': True,
    'deconv_for_upsample': True,
    'deconv_kernel': 3,
    'deconv_pad': 1,
})

yolact_mobilenetv2_concat_32_deconv_even_config = yolact_mobilenetv2_concat_32_config.copy({
    'name': 'yolact_mobilenetv2_concat_32_deconv_even',
    'num_classes': 2,
    'mask_proto_net': [(128, 3, {'padding': 1})] * 3 + [(128, -2, {'padding': 0, 'stride':2}), (128, 3, {'padding': 1})] + [(32, 1, {})],
    'concat_shallow_layer': True,
    'deconv_for_upsample': True,
    'max_size': 480,   
    'deconv_kernel': 2,
    'deconv_pad': 0,
})

yolact_mobilenetv2_concat_32_513_config = yolact_mobilenetv2_concat_32_config.copy({
    'name': 'yolact_mobilenetv2_concat_32_513',
    'num_classes': 2,
    'mask_proto_net': [(128, 3, {'padding': 1})] * 3 + [(None, -257, {}), (128, 3, {'padding': 1})] + [(32, 1, {})],
    'max_size': 513,
    'proto_upsample_size': 129,
})

yolact_efficientnet_b0_config = yolact_base_config.copy({
    'name': 'yolact_efficientnet_b0',
    'num_classes': 2,
    'backbone': efficientnet_backbone.copy({
        'path': 'efficientnet-b0-08094119.pth',
        'args': (['efficientnet-b0']),
        'selected_layers': [4, 5, 7],        
        #'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
    })
})

yolact_efficientnet_b1_config = yolact_base_config.copy({
    'name': 'yolact_efficientnet_b1',
    'num_classes': 2,
    'backbone': efficientnet_backbone.copy({
        'path': 'efficientnet-b1-dbc7070a.pth',
        'args': (['efficientnet-b1']),
        'selected_layers': [4, 5, 7],        
        #'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
    })
})

yolact_efficientnet_b2_config = yolact_base_config.copy({
    'name': 'yolact_efficientnet_b2',
    'num_classes': 2,
    'backbone': efficientnet_backbone.copy({
        'path': 'efficientnet-b2-27687264.pth',
        'args': (['efficientnet-b2']),
        'selected_layers': [4, 5, 7],        
        #'pred_scales': yolact_base_config.backbone.pred_scales,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
        'use_pixel_scales': True,
    })
})

# Default config
cfg = yolact_base_config.copy()

def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

def set_dataset(dataset_name:str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)
    
