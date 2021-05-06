#!/bin/bash

mkdir -p log/detr/
wget -c https://download.pytorch.org/models/resnet50-19c8e357.pth -O ~/.cache/torch/checkpoints/resnet50-19c8e357.pth
T=`date +%m%d%H%M%S`
name=$3
ROOT=.
#cfg=$ROOT/configs/detr/${name}.yaml
g=$(($2<8?$2:8))
COCO=/mnt/lustre/share/DSK/datasets/mscoco2017

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/detr
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=detr_${name} ${SRUN_ARGS} \
python $ROOT/models/detr/detr/main.py \
  --output_dir coco_train --coco_path $COCO --batch_size 4 ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/detr/train.${name}.log.$T