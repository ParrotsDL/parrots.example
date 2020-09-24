#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
from collections import OrderedDict
import json
from matplotlib.image import imread
import pandas as pd
import pickle
import random
import time
import subprocess
import utils.dataloader_utils as dutils

import SimpleITK as sitk
# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.sample_normalization_transforms import RangeTransform
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
# from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates
from experiments.liver_ct_zssy_2d_runet.seg_to_bbox import ConvertSegToBoundingBoxCoordinates


def read_json(file_path):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict


def get_paired_data(root_path, set_name):
    data_dict = read_json('{}/annotations/{}_info.json'.format(root_path, set_name))
    image_list = data_dict['images']
    mask_list = data_dict['annotations']
    paired_dataset = dict()
    area_dict = {i['image_id']: i['area'] for i in mask_list if i['category_id'] == 1}
    for item in image_list:
        if item['id'] in area_dict:
            paired_dataset[item['id']] = {
                'data': os.path.join(root_path, set_name + '_all_ignore10/image', item['file_name']),
                'body': os.path.join(root_path, set_name + '_all_ignore10/mask', item['body_mask_name']),
                'liver': os.path.join(root_path, set_name + '_all_ignore10/mask',
                                      item['liver_mask_name']) if item['liver_mask_name'] is not None else None,
                'lesion': os.path.join(root_path, set_name + '_all_ignore10/mask',
                                       item['lesion_mask_name']) if item['lesion_mask_name'] is not None else None,
                'id': item['id'],
                # 原来是用来表示target的, 但是这里每张图target可以说是一样的
                # 后面会有个平衡采样的过程, 这个target可以用来平衡样本, 比如不同的lesion大小
                # mean 3939 median 1809
                'class_target': [1] if area_dict[item['id']] > 1800 else [0]
            }
    del data_dict
    return paired_dataset


def get_paired_data_3d(root_path, set_name):
    paired_dataset = dict()
    root = os.path.join(root_path, set_name)
    for f in os.listdir(root):
        # if f.find('volume') > -1 and f.find('pv') > -1:
        if f.find('volume') > -1:
            volume_name = f
            volume_path = os.path.join(root, f)
            liver_name = f.replace('volume', 'liver_mask')
            liver_path = os.path.join(root, liver_name)
            if not os.path.exists(liver_path):
                liver_path = None
            lesion_name = f.replace('volume', 'lesion_mask')
            lesion_path = os.path.join(root, lesion_name)
            if not os.path.exists(lesion_path):
                lesion_path = None
            # 一部分是nii文件
            if not os.path.exists(volume_path):
                volume_path = volume_path[:-3]
                assert os.path.exists(volume_path), volume_path

            slice_count = sitk.ReadImage(volume_path).GetSize()[2]
            paired_dataset[volume_name] = {
                'data': volume_path,
                'body': None,
                'liver': liver_path,
                'lesion': lesion_path,
                'count': slice_count,
                'id': volume_name,
            }
    return paired_dataset


def get_paired_data_3d_300newthick(cf):
    from csv import reader
    with open('/data/liver_lesion_seg_data/2020-0322-selected_yuxiang/phase-thickness-checked.csv', 'r', encoding='gb18030') as f:
        r = reader(f)
        data_list = [row for row in r][1:]
    paired_dataset = dict()
    root = '/data/liver_lesion_seg_data/2020-0322-selected_yuxiang/scouting'
    ban_list = os.listdir(cf.image_save_path)

    for row in data_list:
        phase = row[1].lower().strip()
        if 'art' in phase or 'pv' in phase or 'delay' in phase:
        # if f.find('volume') > -1 and f.find('pv') > -1:
        # if f.find('volume') > -1:
            exist_flag = 0
            for name in ban_list:
                if row[0] in name:
                    exist_flag = 1
                    break
            if exist_flag:
                continue
            volume_name = row[0] + '.nii.gz'
            volume_path = os.path.join(root, volume_name)
            # 一部分是nii文件
            if not os.path.exists(volume_path):
                volume_path = volume_path[:-3]
                assert os.path.exists(volume_path), volume_path

            img_itk = sitk.ReadImage(volume_path)
            if img_itk.GetSpacing()[2] < 2:
                print('Ignored thin slice data: ', volume_name)
                continue
            slice_count = img_itk.GetSize()[2]
            paired_dataset[volume_name] = {
                'data': volume_path,
                'body': None,
                'liver': None,
                'lesion': None,
                'count': slice_count,
                'id': row[0] + '_volume_' + phase + '.nii.gz',
            }
    print(len(paired_dataset.keys()))
    return paired_dataset


def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """
    train_data = get_paired_data_3d(cf.root_dir, cf.pp_name)
    val_data = get_paired_data_3d(cf.root_dir, cf.pp_test_name)

    logger.info("data set loaded with: {} train / {} val patients".format(len(train_data.keys()), len(val_data.keys())))
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, do_aug=cf.train_aug)
    batch_gen['val_sampling'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)
    if cf.val_mode == 'val_patient':
        batch_gen['val_patient'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False, rand_sample=False)
        batch_gen['n_val'] = len(val_data.keys())
    else:
        batch_gen['n_val'] = cf.num_val_batches

    return batch_gen


def get_val_generators(cf, logger):
    '''
    get_train_generators 相似, 但是不打乱
    '''
    val_data = get_paired_data_3d(cf.root_dir, cf.pp_test_name)
    batch_gen = dict()
    batch_gen['val_sampling'] = create_data_gen_pipeline(val_data, cf=cf, do_aug=False)

    batch_gen['n_val'] = len(val_data.keys())
    # batch_gen['n_val'] = 10
    logger.info("data set loaded with: {} val patients in {} batchs".format(len(val_data.keys()), batch_gen['n_val']))
    return batch_gen


def get_test_generator(cf, logger):
    """
    每个batch是一个病人, 在test的时候再split
    """
    test_data = get_paired_data_3d(cf.root_dir, cf.pp_test_name)
    # test_data = get_paired_data_3d_300newthick(cf)
    batch_gen = dict()
    batch_gen['val_patient'] = create_data_gen_pipeline(test_data, cf=cf, do_aug=False, rand_sample=False)
    batch_gen['n_test'] = len(test_data.keys())
    return batch_gen


def create_data_gen_pipeline(patient_data, cf, do_aug=True, rand_sample=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(patient_data, batch_size=cf.batch_size, cf=cf, rand_sample=rand_sample)

    # add transformations to pipeline.
    my_transforms = []
    if do_aug:
        # mirror_transform = Mirror(axes=np.arange(2, cf.dim + 2, 1))
        # my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])
        my_transforms.append(spatial_transform)
        range_trasnform = RangeTransform(rnge=cf.image_norm_range)
        my_transforms.append(range_trasnform)
    # else:
    #     my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, dataset_type='liver_lesion_seg'))
    all_transforms = Compose(my_transforms)
    # multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    if cf.server_env:
        # 线程不安全, 中途kill主线程或因某些Error中止时, 线程不会回收
        # 用srun 好像没问题
        multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    else:
        multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    return multithreaded_generator


class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """

    def __init__(self, data, batch_size, cf, rand_sample=True):
        super(BatchGenerator, self).__init__(data, batch_size)

        self.cf = cf
        self.batch_size = batch_size
        self.rand_sample = rand_sample
        self.current_id = 0
        # data在SlimDataloaderBase里
        self.patients = list(self._data.items())
        self.num_patients = len(self.patients)

    def get_arr(self, fl_path):
        return sitk.GetArrayFromImage(sitk.ReadImage(fl_path)).astype(np.float32)

    def norm(self, arr):
        if self.rand_sample:
            down = int(random.uniform(self.cf.image_clip_range[0] - 50, self.cf.image_clip_range[0] + 50))
            up = int(random.uniform(self.cf.image_clip_range[1] - 50, self.cf.image_clip_range[1] + 50))
        else:
            down = self.cf.image_clip_range[0]
            up = self.cf.image_clip_range[1]
        arr = np.clip(arr, down, up)
        arr = (arr - down) / (up - down)
        # assert np.max(arr) == 1 and np.min(arr) == 0
        if np.max(arr) != 1 or np.min(arr) != 0:
            print('\n\n\nArray is not between (0, 1) after min-max norm', np.max(arr), np.min(arr), '\n\n\n')
        if not np.all(self.cf.image_norm_range == [0, 1]):
            arr = arr * (self.cf.image_norm_range[1] - self.cf.image_norm_range[0]) + self.cf.image_norm_range[0]
        return arr.astype(np.float32)

    def get_patient_data_and_seg(self, patient, sl_id_list=None):
        n_c = self.cf.n_3D_context
        data_itk = sitk.ReadImage(patient['data'])

        if sl_id_list is None:
            sl_id_list = [i for i in range(n_c, data_itk.GetSize()[2] - n_c)]
        else:
            assert isinstance(sl_id_list, (list, np.ndarray))
        # data_arr: n, h, w; data: n - 2*n_c, 2*n_c + 1, h, w
        data_arr = sitk.GetArrayFromImage(data_itk).astype(np.float32)
        # norm
        data = self.norm(data_arr)
        data = np.array([data[i - n_c: i + n_c + 1] for i in sl_id_list])

        n, h, w = data_arr.shape
        body_mask = (data_arr >= self.cf.body_mask_threshold).astype(np.float32)
        if patient['liver'] is None:
            liver_mask = np.zeros((n, h, w)).astype(np.float32)
        else:
            liver_mask = self.get_arr(patient['liver'])
        if patient['lesion'] is None:
            lesion_mask = np.zeros((n, h, w)).astype(np.float32)
        else:
            lesion_mask = self.get_arr(patient['lesion'])
        seg = np.stack([body_mask, liver_mask, lesion_mask]).transpose(1, 0, 2, 3)
        seg = np.array([seg[i] for i in sl_id_list])
        pid = [patient['id'] + '_slice' + str(i) for i in sl_id_list]
        class_target = None
        meta = {
            'name': patient['id'],
            'origin': data_itk.GetOrigin(),
            'spacing': data_itk.GetSpacing(),
            'direction': data_itk.GetDirection(),
        }
        return data, seg, pid, class_target, meta

    def generate_train_batch(self):
        # batch_data, batch_segs, batch_pids, batch_targets = [], [], [], []

        if self.rand_sample:
            # 生成一个病人池, 使所包含的slice总数超过 k * batch_size
            k = 2
            # 用来做不重复的采样, 保证生成病人池的时候slice足够
            pat_pool = random.sample(self.patients, min(len(self.patients), self.batch_size))
            # 结构是list of (patient_id, slice_id)
            slice_pool = list()
            while len(slice_pool) < k * self.batch_size:
                # 大池里没病人了
                if len(pat_pool) == 0:
                    break
                cur_id, cur_pat = pat_pool.pop(0)
                n_c = self.cf.n_3D_context
                cur_pat_count = cur_pat['count']
                assert cur_pat_count >= 2 * n_c + 1
                for i in range(n_c, cur_pat_count - n_c):
                    slice_pool.append((cur_id, i))
            # sample得到batch size数量的slice
            slice_list = random.sample(slice_pool, self.batch_size)
            # 变成dict of {patient_id: [slice_id, ]}
            slice_dict = dict()
            for pat_id, sl_id in slice_list:
                if pat_id in slice_dict:
                    slice_dict[pat_id].append(sl_id)
                else:
                    slice_dict[pat_id] = [sl_id]

            # 根据病人和对应的slice id, 读取数据
            data = list()
            seg = list()
            pid = list()
            for pat_id, sl_id_list in slice_dict.items():
                patient = self._data[pat_id]
                d, s, p, _, _ = self.get_patient_data_and_seg(patient, sl_id_list)
                data.append(d)
                seg.append(s)
                pid += p

            # 要求不同病人样本的 h w是一样的, 层厚可以不一样
            data = np.concatenate(data)
            seg = np.concatenate(seg)
            class_target = None
            meta = None

        else:
            # 按病人分batch
            patient = self.patients[self.current_id][1]
            self.current_id += 1

            data, seg, pid, class_target, meta = self.get_patient_data_and_seg(patient)

        return {'data': data, 'seg': seg, 'pid': pid, 'class_target': class_target, 'meta': meta}


def copy_and_unpack_data(logger, pids, fold_dir, source_dir, target_dir):
    start_time = time.time()
    with open(os.path.join(fold_dir, 'file_list.txt'), 'w') as handle:
        for pid in pids:
            handle.write('{}.npy\n'.format(pid))

    subprocess.call('rsync -av --files-from {} {} {}'.format(os.path.join(fold_dir, 'file_list.txt'),
                                                             source_dir, target_dir), shell=True)
    # dutils.unpack_dataset(target_dir)
    copied_files = os.listdir(target_dir)
    logger.info("copying and unpacking data set finsihed : {} files in target dir: {}. took {} sec".format(
        len(copied_files), target_dir, np.round(time.time() - start_time, 0)))


if __name__ == "__main__":
    import utils.exp_utils as utils
    from experiments.liver_ct_zssy_onlylesion.configs import configs

    total_stime = time.time()

    cf = configs()
    logger = utils.get_logger("dev")
    batch_gen = get_train_generators(cf, logger)

    train_batch = next(batch_gen["train"])

    mins, secs = divmod((time.time() - total_stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))
