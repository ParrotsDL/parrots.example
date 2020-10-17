#!/bin/bash

mkdir -p log/pattern_v2_5_sp

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
cfg=$ROOT/configs/pattern_v2_5_sp/${name}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/pattern_v2_5_sp
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 GLOG_logtostderr=-1  OMPI_MCA_btl_smcuda_use_cuda_ipc=0 \
srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=pattern_v2_5_sp_${name} ${SRUN_ARGS} \
python $ROOT/models/pattern_v2_5_sp/tools/dist_train.py \
  --config=$cfg ${EXTRA_ARGS} \
  --now $now $T \
  2>&1 | tee $ROOT/log/pattern_v2_5_sp/train.${name}.log.$T