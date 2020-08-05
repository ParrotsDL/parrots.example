#!/bin/bash

mkdir -p log/pod/

T=`date +%m%d%H%M`
name=$3
ROOT=.
cfg=$ROOT/configs/pod/${name}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/pytorch-object-detection
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=pod_${name} ${SRUN_ARGS} \
python $ROOT/models/pytorch-object-detection/tools/train_val.py \
  --config=$cfg ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/pod/train.${name}.log.$T
