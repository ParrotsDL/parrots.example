#!/bin/bash
set -x

source /usr/local/env/pat_latest

# 容器云网络不太好，设置代理加速下载torchvision的pretrain model, https://download.pytorch.org/models/resnet50-19c8e357.pth 
export http_proxy=http://172.16.1.135:3128/
export https_proxy=http://172.16.1.135:3128/
export HTTP_PROXY=http://172.16.1.135:3128/
export HTTPS_PROXY=http://172.16.1.135:3128/

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

ROOT=.

mkdir -p log/deformable_detr/

now=$(date +"%Y%m%d_%H%M%S")

cfg=$ROOT/configs/deformable_detr/${MODEL_NAME}.yaml
 
pyroot=$ROOT/models/deformable_detr
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH
 
python -u ${pyroot}/main.py \
       --config ${cfg} \
       --with_box_refine \
       --two_stage  ${EXTRA_ARGS} \
       2>&1 | tee ${ROOT}/log/deformable_detr/train.${MODEL_NAME}.log.${now}
