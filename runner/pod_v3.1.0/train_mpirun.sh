#!/bin/bash
source $1
mkdir -p log/pod_v3.1.0/

T=`date +%m%d%H%M%S`
name=$2
ROOT=.
cfg=$ROOT/configs/pod_v3.1.0/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

pyroot=$ROOT/models/pytorch-object-detection-v3.1.0
export PYTHONPATH=$pyroot:$PYTHONPATH
export PYTORCH_VERSION=1.3


python -m pod train \
  --config=$cfg ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/pod_v3.1.0/train.${name}.log.$T
