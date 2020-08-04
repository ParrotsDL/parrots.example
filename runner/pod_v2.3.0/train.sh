#!/bin/bash

mkdir -p log/pod_v2.3.0/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
cfg=$ROOT/configs/pod_v2.3.0/${name}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/pytorch-object-detection-v2.3.0
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=pod_v2.3.0_${name} ${SRUN_ARGS}\
python -m pod train --config=$cfg --display=1 ${EXTRA_ARGS} 2>&1 | tee $ROOT/log/pod_v2.3.0/train.${name}.log.$T
