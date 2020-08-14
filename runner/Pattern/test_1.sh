#!/usr/bin/env bash

ROOT=.
pyroot=$ROOT/models/Pattern
export PYTHONPATH=$pyroot:$PYTHONPATH




array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 


partition=$1
test_gpu=$2
name=$3
config=$ROOT/configs/Pattern/${name}.yaml




python -u $pyroot/tools/run.py \
    --test 1 \
    --config $config \
    --job_name=Pattern_${name} \
    --partition $partition \
    ${EXTRA_ARGS} \


