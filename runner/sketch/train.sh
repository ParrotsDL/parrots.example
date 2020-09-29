#!/bin/bash

mkdir -p log/sketch

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/sketch
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHON_ARGS="python -u models/sketch/tools/train.py configs/sketch/${name}/config.py --launcher=slurm"

set -x
srun -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=sketch_${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/sketch/train.${name}.log.$T

    