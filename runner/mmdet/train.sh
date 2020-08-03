#!/usr/bin/env bash

set -x
ROOT=`pwd`
pyroot=$ROOT/models/mmdetection-pro
export PYTHONPATH=$pyroot:$PYTHONPATH

PARTITION=$1
GPUS=${2:-8}
JOB_NAME=$3
CONFIG=$ROOT/configs/mmdet/${JOB_NAME}.py
WORK_DIR=mmdet/$3
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-"--validate"}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ${pyroot}/tools/train.py ${CONFIG} --work_dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS} \
    2>&1