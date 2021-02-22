dataset_type = 'MSDDataset'
data_root =  '/mnt/lustre/share_data/parrots_model_data/sensemedical/Task01_BrainTumour'
ceph_data_root = 's3://parrots_model_data/sensemedical'
fold = 0
train_transformer = [
    dict(type='RandPosCrop', spatial_size= [112,112,112], pos_ratio = 0.8),
    dict(type='RandAffine',rotate = [20,20,20], spatial_size = [112,112,112]),
    dict(type='RandMirror'),
    dict(type='RandAdjustContrast', gamma = [0.5, 4.5]),
    dict(type='ToTensor')
]
val_transformer = [
    # dict(type='CenterCrop', crop_size = [112,112,112]),
    dict(type='ToTensor')
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        fold=fold,
        data_root=data_root,
        transformer=train_transformer,
        phase='train'),
    val=dict(
        type=dataset_type,
        fold=fold,
        data_root=data_root,
        transformer=val_transformer,
        phase='val'),
)

model= dict(
    type='Segmentor',
    network=dict(
        type='UNet3D',
        in_channels=4,
        num_classes=4,
        f_maps=32,
        num_levels=4,
        norm='bn',
        act='relu'),
    losses=[
        dict(type='DiceLoss', act='softmax'),
        dict(type='CrossEntropy')],
    metrics=dict(
        class_names=['BackGround','NET','ED','ET'],
        metric_types = [dict(type='DiceMetric', exclude_backgroud=True, act='softmax')])
)

evaluation = dict(interval=1, evaluator_type='slidingwindow', eval_kwargs=dict(roi_size=[112,112,112], sw_batch_size=4, overlap=0.25))
# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='fixed', by_epoch=True)
# runtime settings
max_epochs = 200
checkpoint_config = dict(by_epoch=True, interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]#,('val', 1)]
cudnn_benchmark = True
work_dir = './work_dirs/Task01_BrainTumour_fold_{}'.format(fold)
