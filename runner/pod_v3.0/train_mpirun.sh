#!/bin/bash
source $1

mkdir -p log/pod_v3.0
now=$(date +"%Y%m%d_%H%M%S")
set -x

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

ROOT=.
pyroot=$ROOT/models/pytorch-object-detection-v3.0/

name=$2
cfg=$ROOT/configs/pod_v3.0/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
g=$(($2<8?$2:8))
export PYTHONPATH=$pyroot:$PYTHONPATH

python -m pod train \
  --config=${cfg} \
  --display=1 \
   ${EXTRA_ARGS} \
  2>&1 | tee log/pod_v3.0/train_${name}.log-$now
