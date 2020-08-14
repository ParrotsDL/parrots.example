#!/usr/bin/env bash
mkdir -p log/Pattern
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/Pattern
export PYTHONPATH=$pyroot:$PYTHONPATH



partition=$1
train_gpu=$2
name=$3
config=$ROOT/configs/Pattern/${name}.yaml


array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
#echo --train 1 --config $config --job_name=Pattern_${name}  --partition $partition --train_gpu $train_gpu --extra_args ${EXTRA_ARGS} --srun_args ${SRUN_ARGS} 

python -u $pyroot/tools/run.py \
    --train 1 \
    --config $config \
    --job_name=Pattern_${name} \
    --partition $partition \
    --train_gpu $train_gpu \
    ${EXTRA_ARGS} \
    2>&1 | tee log/Pattern/train_${name}.log-$now

