#!/bin/sh
NODE_NUM=1
GPU_NUM=8
MODEL_NAME=resnet
TOTAL_NUM=$((${NODE_NUM} * ${GPU_NUM}))

export NVIDIA_DRIVER_CAPABILITIES="compute,utility"
export NVIDIA_REQUIRE_CUDA="cuda>=8.0"
export MV2_ENABLE_AFFINITY="0"

mpirun --allow-run-as-root \
    --hostfile ./mpirun_hostfile \
    --np ${TOTAL_NUM} \
    --npernode ${GPU_NUM} \
    bash mpirun_main.sh \
    ${MODEL_NAME}