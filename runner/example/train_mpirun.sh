#!/bin/bash
set -x
source $1

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

ROOT=.

mkdir -p log/example/

now=$(date +"%Y%m%d_%H%M%S")

cfg=$ROOT/configs/example/${name}.yaml
 
pyroot=$ROOT/models/parrots.example
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH
 
python -u $ROOT/models/parrots.example/models/imagenet/main.py --config ${cfg} \
        ${EXTRA_ARGS} \
        2>&1 | tee log/example/train_${name}.log-$now
