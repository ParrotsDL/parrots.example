#!/usr/bin/env bash

echo Usage: sh slurm_train.sh [partition] [job_name] [number_cards] [number_nodes] [other_params]

set -x

ROOT=.
pyroot=$ROOT/models/sketch1.4
export PYTHONPATH=$pyroot:$PYTHONPATH

PARTITION=$1
GPUS=$2
name=$3
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
g=$(($2<8?$2:8))

#!/bin/bash
mkdir -p log/sketch1.4
T=`date +%m%d%H%M%S`
CURRENT_DIR=$PWD
MAIN_DIR=$(cd "$(dirname "$0")";pwd)
cd $CURRENT_DIR



case $GPUS in [1-9]|[1-9][0-9]*) echo "this job use $GPUS cards";; *) echo "$GPUS is not a positive number"; exit;; esac

SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}
echo $PY_ARGS

py_args="python -u $pyroot/tools/train.py configs/sketch1.4/${name}/config.py --work-dir=$PWD/sketch1.4_work-dirs/$name  ${PY_ARGS} --launcher=slurm"

OMP_NUM_THREADS=1
srun -p ${PARTITION} \
     --ntasks=${GPUS} \
     --gres=gpu:${g} \
    --ntasks-per-node=${g} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --job-name=sketch1.4_$name \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ${py_args} \
    ${PY_ARGS} \
    2>&1 | tee $ROOT/log/sketch1.4/train.${name}.log.$T
