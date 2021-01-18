#!/bin/bash

mkdir -p log/Multi_organ_seg_HR

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/Multi_organ_seg_HR
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHON_ARGS="python -u models/Multi_organ_seg_HR/train.py --config configs/Multi_organ_seg_HR/config_thoracic.yaml"

set -x
srun -p $1 --gres gpu:$g --job-name=Multi_organ_seg_HR_${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/Multi_organ_seg_HR/train.${name}.log.$T
