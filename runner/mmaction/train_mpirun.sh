#!/bin/bash
set -x

source /usr/local/env/pat_latest

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

MODEL_NAME=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

mkdir -p log/mmaction

T=`date +%m%d%H%M%S`
name=${MODEL_NAME}
ROOT=.

pyroot=$ROOT/models/mmaction
export PYTHONPATH=$pyroot:$PYTHONPATH

PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/${name}.py --launcher=mpi --validate"

set -x
$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmaction/train.${name}.log.$T