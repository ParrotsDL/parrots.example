#!/bin/bash
set -x

source $1

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

MODEL_NAME=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

mkdir -p log/sketch

T=`date +%m%d%H%M%S`
name=${MODEL_NAME}
ROOT=.

pyroot=$ROOT/models/sketch
export PYTHONPATH=$pyroot:$PYTHONPATH

PYTHON_ARGS="python -u models/sketch/tools/train.py configs/sketch/${name}/config.py --launcher=mpi"

set -x
$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/sketch/train.${name}.log.$T