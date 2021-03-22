#!/usr/bin/env bash

source $1

mkdir -p log/Pattern
now=$(date +"%Y%m%d_%H%M%S")

set -x
ROOT=.
pyroot=$ROOT/models/Pattern
export PYTHONPATH=$pyroot:$PYTHONPATH

export pattern_container_job=1

# 容器的pod每次都要复制一次，pod都是新起的。

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
SRUN_ARGS=${SRUN_ARGS:-""}


partition="place-holder"
train_gpu=0
config=$2


SRUN_ARGS=${SRUN_ARGS}  sh runner/Pattern/train_1.sh $partition $train_gpu   $config  ${EXTRA_ARGS} 
SRUN_ARGS=${SRUN_ARGS} sh runner/Pattern/test_1.sh  $partition 1  $config   ${EXTRA_ARGS} 

