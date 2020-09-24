#!/bin/bash

mkdir -p log/encoder/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/encoder
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=pod_${name} ${SRUN_ARGS} \
python $ROOT/models/encoder/train_benign.py \
  ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/encoder/train.${name}.log.$T
