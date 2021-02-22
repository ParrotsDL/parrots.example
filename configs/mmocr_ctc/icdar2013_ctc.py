_base_ = [
    '../../models/mmocr_ctc/configs/_base_/schedules/schedule_adadelta_8e.py',
    '../../models/mmocr_ctc/configs/_base_/default_runtime.py'
]

vocab_file = "models/mmocr_ctc/dict/academic_dict.txt"
label_convertor = dict(
    type='CTCLabelConvertor', vocab_file=vocab_file, with_unknown=True)

model = dict(
    type='CTCRecognizer',
    backbone=dict(
        type='IntraResNet31',
        channels=[32, 64, 128, 256, 512, 512],
        layers=[1, 2, 5, 3]),
    recog_head=dict(type='IntraCTCHead', rnn_flag=False),
    label_convertor=label_convertor)

train_cfg = None
test_cfg = None

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeOCR', height=32, width=256, keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeOCR', height=32, width=256, keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'valid_ratio']),
]
dataset_type = 'OCRRecogDataset'
#train_img_prefix1 = 'data/icdar_2013/train/'
train_img_prefix1 = '/mnt/lustre/share_data/parrots_model_data/mmocr/data/ctc/icdar_2013/train/'
ceph_train_img_prefix1 = 's3://parrots_model_data/mmocr/data/ctc/icdar_2013/train/'
train_anno_file1 = train_img_prefix1 + 'train_label.txt'
train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix1,
    ann_file=train_anno_file1,
    loader=dict(
        type='TextLoader',
        repeat=5,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            split_by=[' '])),
    pipeline=train_pipeline,
    test_mode=False)

test_img_prefix1 = '/mnt/lustre/share_data/parrots_model_data/mmocr/data/ctc/icdar_2013/test/'
ceph_test_img_prefix1 = 's3://parrots_model_data/mmocr/data/ctc/icdar_2013/test/'
test_anno_file1 = test_img_prefix1 + 'test_label.txt'
test1 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix1,
    ann_file=test_anno_file1,
    loader=dict(
        type='TextLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            split_by=[' '])),
    pipeline=test_pipeline,
    test_mode=True)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train1]),
    val=dict(type='ConcatDataset', datasets=[test1]),
    test=dict(type='ConcatDataset', datasets=[test1]))

evaluation = dict(interval=1, metric='acc')
