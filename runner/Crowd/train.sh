#!/usr/bin/env bash

mkdir -p log/Crowd/
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/Crowd
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/Crowd/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

GLOG_vmodule=MemcachedClient=-1 
srun -p $1 \
    --mpi=pmi2 \
    -n$2 \
    --gres=gpu:$g \
    --ntasks-per-node=$g \
    --kill-on-bad-exit=1 \
    --job-name=Crowd_${name} \
        ${SRUN_ARGS} \
python -u models/Crowd/main.py \
       --config=${cfg} \
       ${EXTRA_ARGS} \
    2>&1 | tee log/Crowd/Crowd_${name}.log-$now