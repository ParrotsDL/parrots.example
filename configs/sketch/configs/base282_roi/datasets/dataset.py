

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        pipeline = [],
        ),
    val=dict(
        type='CustomDataset',
        ),
    )
