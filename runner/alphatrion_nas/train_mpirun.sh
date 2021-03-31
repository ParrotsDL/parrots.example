#!/bin/bash
source $1
mkdir -p log/alphatrion_nas/

T=`date +%m%d%H%M%S`
name=$2
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
ROOT=.
pyroot=$ROOT/models/alphatrion_nas
alphatrion_deps=$ROOT/models/alphatrion/alphatrion
export PYTHONPATH=$pyroot:$alphatrion_deps:$PYTHONPATH

if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

case $name in
    "super_resnet_range1")
      PYTHON_ARGS="python -u models/alphatrion_nas/code/super_learn.py \
      -cfg configs/alphatrion_nas/base.yaml \
      configs/alphatrion_nas/super_resnet_base.yaml \
      configs/alphatrion_nas/augment_randaugment.yaml \
      configs/alphatrion_nas/super_learn_part_train.yaml \
      --search_space super_resnet_range1 \
      --RA.M 11 --all_num_epochs.super_learn 20 \
      --local_batch_size 128 \
      --lr 0.8 \
      --fp 32 --mixed.train false --mixed.valid false --mixed.adabn false \
      --resume ''"
    ;;
    "super_resnet_range1_fp16")
      PYTHON_ARGS="python -u models/alphatrion_nas/code/super_learn.py \
      -cfg configs/alphatrion_nas/base.yaml \
      configs/alphatrion_nas/super_resnet_base.yaml \
      configs/alphatrion_nas/augment_randaugment.yaml \
      configs/alphatrion_nas/super_learn_part_train.yaml \
      --search_space super_resnet_range1 \
      --RA.M 11 --all_num_epochs.super_learn 20 \
      --local_batch_size 128 \
      --lr 0.8 \
      --fp 16 --mixed.train true --mixed.valid true --mixed.adabn false --mixed.loss_scale 512 \
      --resume ''"
    ;;
    "super_resnet_range1_benchmark")
      PYTHON_ARGS="python -u models/alphatrion_nas/code/super_learn.py \
      -cfg configs/alphatrion_nas/base.yaml \
      configs/alphatrion_nas/super_resnet_base.yaml \
      configs/alphatrion_nas/augment_randaugment.yaml \
      configs/alphatrion_nas/super_learn_part_train.yaml \
      --search_space super_resnet_range1 \
      --RA.M 11 --all_num_epochs.super_learn 1 \
      --local_batch_size 64 \
      --lr 0.8 \
      --fp 32 --mixed.train false --mixed.valid false --mixed.adabn false \
      --resume ''"
    ;;
    "super_resnet_range1_fp16_benchmark")
      PYTHON_ARGS="python -u models/alphatrion_nas/code/super_learn.py \
      -cfg configs/alphatrion_nas/base.yaml \
      configs/alphatrion_nas/super_resnet_base.yaml \
      configs/alphatrion_nas/augment_randaugment.yaml \
      configs/alphatrion_nas/super_learn_part_train.yaml \
      --search_space super_resnet_range1 \
      --RA.M 11 --all_num_epochs.super_learn 1 \
      --local_batch_size 128 \
      --lr 0.8 \
      --fp 16 --mixed.train true --mixed.valid true --mixed.adabn false --mixed.loss_scale 512 \
      --resume ''"
    ;;
    *)
      echo "invalid $name, only support: [super_resnet_range1, super_resnet_range1_fp16]"
      exit 1
      ;; 
esac
set -x

$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/alphatrion_nas/train.${name}.log.$T
