#!/bin/bash
set -x

source $1 


MODEL_NAME=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

mkdir -p log/mmediting

T=`date +%m%d%H%M%S`

ROOT=.
pyroot=$ROOT/models/mmediting
export PYTHONPATH=$pyroot:$PYTHONPATH
export PARROTS_POOL_DATALOADER=1
export PYTHONPATH=/mnt/lustre/share/memcached:$PYTHONPATH



## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi


# 避免mm系列重复打印
export PARROTS_DEFAULT_LOGGER=FALSE

PYTHON_ARGS="python -u models/mmediting/tools/train.py configs/mmediting/${MODEL_NAME}.py --launcher=mpi"

set -x

$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmediting/train.${MODEL_NAME}.log.$T

    
