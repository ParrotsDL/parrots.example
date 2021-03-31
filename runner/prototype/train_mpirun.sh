#!/bin/bash
set -x
export PYTHONPATH=/mnt/lustre/share/pymc/py3/:$PYTHONPATH
source $1

# 多机多卡的训练脚本
export PYTHONPATH=/senseparrots/python:/PAPExtension:/mnt/lustre/share/memcached:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/libmemcached/lib:$LD_LIBRARY_PATH
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

name=$2
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

mkdir -p log/prototype
now=$(date +"%Y%m%d_%H%M%S")
ROOT=.
cfg=$ROOT/configs/prototype/${name}.yaml
#g=$(($2<8?$2:8))

pyroot=$ROOT/models/prototype
export PYTHONPATH=$pyroot:$PYTHONPATH


    python -u -m prototype.solver.cls_solver --config ${cfg} \
    ${EXTRA_ARGS} \
    2>&1 | tee log/prototype/train_${name}.log-$now
