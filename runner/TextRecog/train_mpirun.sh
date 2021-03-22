#!/bin/bash
set -x

source $1

MODEL_NAME=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}


## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

mkdir -p log/TextRecog/

T=`date +%m%d%H%M%S`
name=${MODEL_NAME}
ROOT=.
cfg=$ROOT/configs/TextRecog/${name}_bn1.yaml

pyroot=$ROOT/models/TextRecog
export PYTHONPATH=$pyroot:$PYTHONPATH


python -u models/TextRecog/tools/train_val.py \
  --config=$cfg --phase train --launcher mpi  ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/TextRecog/train.${name}.log.$T
