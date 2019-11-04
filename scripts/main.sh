#!/bin/sh
/mnt/lustre/share/platform/dep/openmpi-2.1.6-cuda9.0/bin/mpirun \
    --allow-run-as-root \
    --hostfile ./mpirun_hostfile \
    --np 2 \
    --npernode 2 \
    bash mpirun_main.sh \
    resnet \
    --max_epoch 10
