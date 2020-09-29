#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/26 下午5:27
# @Author  : huangyechong
# @Site    : 
# @File    : seg_to_bbox.py
# @Software: PyCharm

import random
import numpy as np
from scipy.ndimage.measurements import label as lb
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class ConvertSegToBoundingBoxCoordinates(AbstractTransform):
    """ Converts segmentation masks into bounding box coordinates.
    """

    def __init__(self, dim, dataset_type=None, get_rois_from_seg_flag=False, class_specific_seg_flag=False):
        self.dim = dim
        self.dataset_type = dataset_type
        self.get_rois_from_seg_flag = get_rois_from_seg_flag
        self.class_specific_seg_flag = class_specific_seg_flag

    def __call__(self, **data_dict):
        if self.dataset_type is None:
            raise NotImplementedError
            # data_dict = convert_seg_to_bounding_box_coordinates(data_dict, self.dim, self.get_rois_from_seg_flag, class_specific_seg_flag=self.class_specific_seg_flag)
        elif self.dataset_type == 'liver_lesion_seg':
            data_dict = convert_seg_to_bounding_box_coordinates_liver(data_dict, self.dim)
        return data_dict


# def convert_seg_to_bounding_box_coordinates(data_dict, dim, get_rois_from_seg_flag=False,
#                                             class_specific_seg_flag=False):
#
#     '''
#     This function generates bounding box annotations from given pixel-wise annotations.
#     :param data_dict: Input data dictionary as returned by the batch generator.
#     :param dim: Dimension in which the model operates (2 or 3).
#     :param get_rois_from_seg: Flag specifying one of the following scenarios:
#     1. A label map with individual ROIs identified by increasing label values, accompanied by a vector containing
#     in each position the class target for the lesion with the corresponding label (set flag to False)
#     2. A binary label map. There is only one foreground class and single lesions are not identified.
#     All lesions have the same class target (foreground). In this case the Dataloader runs a Connected Component
#     Labelling algorithm to create processable lesion - class target pairs on the fly (set flag to True).
#     :param class_specific_seg_flag: if True, returns the pixelwise-annotations in class specific manner,
#     e.g. a multi-class label map. If False, returns a binary annotation map (only foreground vs. background).
#     :return: data_dict: same as input, with additional keys:
#     - 'bb_target': bounding box coordinates (b, n_boxes, (y1, x1, y2, x2, (z1), (z2)))
#     - 'roi_labels': corresponding class labels for each box (b, n_boxes, class_label)
#     - 'roi_masks': corresponding binary segmentation mask for each lesion (box). Only used in Mask RCNN. (b, n_boxes, y, x, (z))
#     - 'seg': now label map (see class_specific_seg_flag)
#     '''
#
#     bb_target = []
#     roi_masks = []
#     roi_labels = []
#     out_seg = np.copy(data_dict['seg'])
#     for b in range(data_dict['seg'].shape[0]):
#
#         p_coords_list = []
#         p_roi_masks_list = []
#         p_roi_labels_list = []
#
#         if np.sum(data_dict['seg'][b] != 0) > 0:
#             if get_rois_from_seg_flag:
#                 clusters, n_cands = lb(data_dict['seg'][b])
#                 class_targets = data_dict['class_target'][b]
#                 # 统一为list 因为array和list乘法不一样
#                 if isinstance(class_targets, list):
#                     pass
#                 elif isinstance(class_targets, np.ndarray):
#                     class_targets = class_targets.tolist()
#                 else:
#                     raise TypeError('Unsupported class target: {}'.format(type(class_targets)))
#                 data_dict['class_target'][b] = class_targets * n_cands
#             else:
#                 n_cands = int(np.max(data_dict['seg'][b]))
#                 clusters = data_dict['seg'][b]
#
#             rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])  # separate clusters and concat
#             for rix, r in enumerate(rois):
#                 if np.sum(r != 0) > 0:  # check if the lesion survived data augmentation
#                     seg_ixs = np.argwhere(r != 0)
#                     coord_list = [np.min(seg_ixs[:, 1]) - 1, np.min(seg_ixs[:, 2]) - 1, np.max(seg_ixs[:, 1]) + 1,
#                                   np.max(seg_ixs[:, 2]) + 1]
#                     if dim == 3:
#                         coord_list.extend([np.min(seg_ixs[:, 3]) - 1, np.max(seg_ixs[:, 3]) + 1])
#
#                     p_coords_list.append(coord_list)
#                     p_roi_masks_list.append(r)
#                     # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
#                     # also patient wide, this assignment is not dependent on patch occurrances.
#                     p_roi_labels_list.append(data_dict['class_target'][b][rix] + 1)
#
#                 if class_specific_seg_flag:
#                     out_seg[b][data_dict['seg'][b] == rix + 1] = data_dict['class_target'][b][rix] + 1
#
#             if not class_specific_seg_flag:
#                 out_seg[b][data_dict['seg'][b] > 0] = 1
#
#             bb_target.append(np.array(p_coords_list))
#             roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
#             roi_labels.append(np.array(p_roi_labels_list))
#
#
#         else:
#             bb_target.append([])
#             roi_masks.append(np.zeros_like(data_dict['seg'][b])[None])
#             roi_labels.append(np.array([-1]))
#
#     if get_rois_from_seg_flag:
#         data_dict.pop('class_target', None)
#
#     data_dict['bb_target'] = np.array(bb_target)
#     data_dict['roi_masks'] = np.array(roi_masks)
#     data_dict['roi_labels'] = np.array(roi_labels)
#     # data_dict['roi_labels'] = roi_labels
#     data_dict['seg'] = out_seg
#
#     return data_dict

def convert_seg_to_bounding_box_coordinates_liver(data_dict, dim):

    '''
    This function generates bounding box annotations from given pixel-wise annotations.
    :param data_dict: Input data dictionary as returned by the batch generator.
    :param dim: Dimension in which the model operates (2 or 3).
    专门为liver lesion seg做的数据集
    分类为 body: [0], liver [1] lesion [2]
    lesion可能有多个
    通常地 lesion一定在liver内, liver一定在body内
    :return: data_dict: same as input, with additional keys:
    - 'bb_target': bounding box coordinates (b, n_boxes, (y1, x1, y2, x2, (z1), (z2)))
    - 'roi_labels': corresponding class labels for each box (b, n_boxes, class_label)
    - 'roi_masks': corresponding binary segmentation mask for each lesion (box). Only used in Mask RCNN. (b, n_boxes, y, x, (z))
    - 'seg': now label map (see class_specific_seg_flag)
    '''

    bb_target = []
    roi_masks = []
    roi_labels = []
    out_seg = np.copy(data_dict['seg'])
    for b in range(data_dict['seg'].shape[0]):

        p_coords_list = []
        p_roi_masks_list = []
        p_roi_labels_list = []

        if np.sum(data_dict['seg'][b] != 0) > 0:
            class_targets = list()
            for ch in range(data_dict['seg'].shape[1]):
                # current channel of seg
                curr_ch_seg = data_dict['seg'][b][ch]
                # body层和liver层应当视为整体
                if ch > 1:
                    clusters, n_cands = lb(curr_ch_seg)
                else:
                    clusters = curr_ch_seg
                    n_cands = 1
                if n_cands > 0:
                    class_targets.extend([ch] * n_cands)

                    # separate clusters and concat
                    rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])
                    for rix, r in enumerate(rois):
                        if np.sum(r != 0) > 0:  # check if the lesion survived data augmentation
                            seg_ixs = np.argwhere(r != 0)
                            # 原版是 1, 384, 384; 这里是384, 384
                            coord_list = [np.min(seg_ixs[:, 0]) - 1, np.min(seg_ixs[:, 1]) - 1,
                                          np.max(seg_ixs[:, 0]) + 1, np.max(seg_ixs[:, 1]) + 1]
                            if dim == 3:
                                coord_list.extend([np.min(seg_ixs[:, 2]) - 1, np.max(seg_ixs[:, 2]) + 1])

                            p_coords_list.append(coord_list)
                            p_roi_masks_list.append(r[np.newaxis])
                            # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
                            # also patient wide, this assignment is not dependent on patch occurrances.
                            p_roi_labels_list.append(ch + 1)


                # out_seg[b][data_dict['seg'][b] == rix + 1] = data_dict['class_target'][b][rix] + 1
                # 把都是1的mask, 变成body 1, liver 2, lesion 3
                out_seg[b][ch] = out_seg[b][ch] * (ch + 1)


            bb_target.append(np.array(p_coords_list))
            roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
            roi_labels.append(np.array(p_roi_labels_list))


        else:
            bb_target.append([])
            roi_masks.append(np.zeros_like(data_dict['seg'][b])[None])
            roi_labels.append(np.array([-1]))

    data_dict.pop('class_target', None)

    data_dict['bb_target'] = np.array(bb_target)
    data_dict['roi_masks'] = np.array(roi_masks)
    data_dict['roi_labels'] = np.array(roi_labels)
    # data_dict['roi_labels'] = roi_labels
    data_dict['seg'] = out_seg

    return data_dict