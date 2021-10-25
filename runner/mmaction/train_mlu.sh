#!/bin/bash

mkdir -p log/mmaction

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/mmaction
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

# 避免mm系列重复打印
export PARROTS_DEFAULT_LOGGER=FALSE

PYTHON_ARGS="python -u models/mmaction/tools/train.py --config configs/mmaction/${name}.py --launcher=slurm --validate"

set -x
srun -p $1 -n$2 --gres mlu:$g --ntasks-per-node $g --job-name=mmaction_${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmaction/train.${name}.log.$T

    
