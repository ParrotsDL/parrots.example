#!/bin/bash

mkdir -p log/mouth/

T=`date +%m%d%H%M`
name=$3
ROOT=.
cfg=$ROOT/configs/mouth/${name}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/mouth_inpaint_network
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g --job-name=mouth_${name} ${SRUN_ARGS}\
python $ROOT/models/mouth_inpaint_network/train.py \
  $cfg  log/mouth/ ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/mouth/train.mouth_${name}.log.$T

