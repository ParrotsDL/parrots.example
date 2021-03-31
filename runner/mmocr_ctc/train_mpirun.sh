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
WORKING_ROOT=$PWD

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
    rm -rf /home/${USER}/.ssh
    cp -r /mnt/lustre/${USER}/.ssh/ /home/${USER}/.ssh/
    cd /home/${USER}
    git clone -b pat_v2.4.0 git@gitlab.sz.sensetime.com:parrotsDL-sz/mmdetection.git mmdet
    cd $WORKING_ROOT
fi

ROOT=.

mkdir -p log/mmocr_ctc/

now=$(date +"%Y%m%d_%H%M%S")

cfg=$ROOT/configs/mmocr_ctc/${MODEL_NAME}.py

pyroot=$ROOT/models/mmocr_ctc/
export PYTHONPATH=$pyroot:$PYTHONPATH

while [ ! -d "/home/${USER}/mmdet/" ]
do
  sleep 1s
done
cd /home/${USER}/mmdet/
export PYTHONPATH=$PWD:$PYTHONPATH 
cd $WORKING_ROOT

python -u $ROOT/models/mmocr_ctc/tools/train.py \
  $cfg  --work-dir=$ROOT/models/mmocr_ctc/work_dir/${MODEL_NAME} --launcher="mpi" ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/mmocr_ctc/train.${MODEL_NAME}.log.$now
  