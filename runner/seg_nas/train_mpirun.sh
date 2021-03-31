#!/bin/bash
set -x

source $1

# 多机多卡的训练脚本
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

MODEL_NAME=$2
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

ROOT=.

mkdir -p log/Light_Nas_zpzhang/

now=$(date +"%Y%m%d_%H%M%S")

cfg=$ROOT/configs/seg_nas/${MODEL_NAME}/config.yaml

pyroot=$ROOT/models/Light_Nas_zpzhang
export PYTHONPATH=$pyroot:$PYTHONPATH

step=evolution

python -u $ROOT/models/Light_Nas_zpzhang/main.py \
         --config=$cfg ${EXTRA_ARGS} \
         --step=$step \
         2>&1 | tee $ROOT/log/Light_Nas_zpzhang/train.${name}.${step}.log.$now
