
task_type='cls'
region_type='head'

#data
samples_per_gpu=64
workers_per_gpu=8

#ckpt & log
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='PaviLoggerHook',
            add_graph=False, add_last_ckpt=True),
    ])
# yapf:enable
dist_params = dict(backend='nccl', port=25589)
log_level = 'INFO'
load_from = None
resume_flag = False
workflow = [('train', 1)]
evaluation = dict(interval=2)
seed=2020

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    gamma=0.5,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[50, 60, 70, 80, 90])
total_epochs = 100

