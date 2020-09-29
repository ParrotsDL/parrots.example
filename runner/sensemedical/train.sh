#!/bin/bash

mkdir -p log/sensemedical/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
cfg=$ROOT/configs/sensemedical/${name}.py
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/sensemedical
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=sensemedical_${name} ${SRUN_ARGS} \
python3 $ROOT/models/sensemedical/tools/train_seg.py \
  $cfg  --launcher="slurm" ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/sensemedical/train.${name}.log.$T
