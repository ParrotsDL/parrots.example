#!/bin/bash
set -x
 
source /usr/local/env/pat_latest
 
export PYTHONPATH=/senseparrots/python:/PAPExtension:/mnt/lustre/share/memcached:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/libmemcached/lib:$LD_LIBRARY_PATH
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"
 
MODEL_NAME=$1
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
 
## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi
 
ROOT=.
 
mkdir -p log/mmpose/
 
now=$(date +"%Y%m%d_%H%M%S")

cfg=$ROOT/configs/mmpose/${MODEL_NAME}.py
 
pyroot=$ROOT/models/mmpose
mmcvroot=$ROOT/mmcv
export PYTHONPATH=${pyroot}:${mmcvroot}:$PYTHONPATH

python -u $ROOT/models/mmpose/tools/train.py \
       $cfg \
       --work-dir=${work_dir} \
       --launcher="mpi" \
       ${EXTRA_ARGS} \
    2>&1|tee log/mmpose/train_${MODEL_NAME}.log-$now
