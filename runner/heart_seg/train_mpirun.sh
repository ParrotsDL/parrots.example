#!/bin/bash
set -x
source /usr/local/env/pat_latest

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

mkdir -p log/heart_seg/
BS=`expr 1 \* $OMPI_COMM_WORLD_SIZE`

now=$(date +"%Y%m%d_%H%M%S")
ROOT=.

cfg=$ROOT/configs/heart_seg/${MODEL_NAME}.yaml

pyroot=$ROOT/models/heart_seg
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH

python ${pyroot}/code/trainSegVNet.py --config ${cfg} \
    --fold 0 --lr 0.001 --batch-size ${BS} --num-workers ${BS} --base-features 16 \
    --loss cross_entropy --val-interval 5 --test-interval 10\
    --output-dir "${pyroot}/OutSeg" --epochs 400 ${EXTRA_ARGS} \
    2>&1 | tee ${ROOT}/log/heart_seg/train.${MODEL_NAME}.log.${now}
