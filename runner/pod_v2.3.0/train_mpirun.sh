#!/bin/bash
set -x

source /usr/local/env/pat_latest

MODEL_NAME=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

mkdir -p log/pod_v2.3.0/

T=`date +%m%d%H%M%S`

ROOT=.
cfg=$ROOT/configs/pod_v2.3.0/${MODEL_NAME}.yaml

pyroot=$ROOT/models/pytorch-object-detection-v2.3.0
export PYTHONPATH=$pyroot:$PYTHONPATH

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

python -m pod train --config=$cfg --display=1 ${EXTRA_ARGS} 2>&1 | tee $ROOT/log/pod_v2.3.0/train.${MODEL_NAME}.log.$T
