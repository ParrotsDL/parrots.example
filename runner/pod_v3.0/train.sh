#!/bin/bash
mkdir -p log/pod_v3.0
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/pytorch-object-detection-v3.0/
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
cfg=$ROOT/configs/pod_v3.0/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
g=$(($2<8?$2:8))
pyroot=$ROOT/models/pytorch-object-detection-v3.0/
export PYTHONPATH=$pyroot:$PYTHONPATH

srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g  \
--job-name=pod_v3.0_${name} ${SRUN_ARGS}\
python -m pod train \
  --config=${cfg} \
  --display=1 \
   ${EXTRA_ARGS} \
  2>&1 | tee log/pod_v3.0/train_${name}.log-$now
