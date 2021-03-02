#!/bin/bash
set -x

source pat_latest

# 多机多卡的训练脚本
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

MODEL_NAME=$1
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

ROOT=.

mkdir -p log/Crowd/

now=$(date +"%Y%m%d_%H%M%S")

cfg=$ROOT/configs/Crowd/${MODEL_NAME}.yaml
pyroot=$ROOT/models/Crowd
export PYTHONPATH=$pyroot:$PYTHONPATH

python -u models/Crowd/main.py \
       --config=${cfg} \
       ${EXTRA_ARGS} \
    2>&1 | tee log/Crowd/Crowd_${MODEL_NAME}.log.$now
