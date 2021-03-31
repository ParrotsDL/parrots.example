#!/bin/bash
set -x
source $1

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

mkdir -p log/mmtrack/

now=$(date +"%Y%m%d_%H%M%S")
 
pyroot=$ROOT/models/mmtrack
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH
 
sh runner/mmtrack/run_train_mpirun.sh ${MODEL_NAME} ${EXTRA_ARGS}
# sh runner/mmtrack/run_test_mpirun.sh ${MODEL_NAME} ${EXTRA_ARGS} 
