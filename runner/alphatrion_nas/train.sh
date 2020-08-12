#!/bin/bash

mkdir -p log/alphatrion_nas/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/alphatrion_nas
alphatrion_deps=$ROOT/models/alphatrion/alphatrion
export PYTHONPATH=$pyroot:$alphatrion_deps:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

case $name in
    "super_resnet_range1")
      PYTHON_ARGS="python -u models/alphatrion_nas/code/super_learn.py \
      -cfg configs/alphatrion_nas/base.yaml \
      configs/alphatrion_nas/super_resnet_base.yaml \
      configs/alphatrion_nas/augment_randaugment.yaml \
      configs/alphatrion_nas/super_learn_part_train.yaml \
      --search_space super_resnet_range1 \
      --RA.M 11 --all_num_epochs.super_learn 20 \
      --local_batch_size 128 --num_gpus $2 \
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
      --local_batch_size 128 --num_gpus $2 \
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
      --local_batch_size 64 --num_gpus $2 \
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
      --local_batch_size 128 --num_gpus $2 \
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
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=alphatrion_nas_${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/alphatrion_nas/train.${name}.log.$T
