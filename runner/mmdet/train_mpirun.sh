#!/bin/bash

set -x

source /usr/local/env/pat_latest

export PYTHONPATH=/senseparrots/python:/PAPExtension:/mnt/lustre/share/memcached:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/libmemcached/lib:$LD_LIBRARY_PATH
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"
export PYTORCH_VERSION=1.4

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

mkdir -p log/mmdet/

now=$(date +"%Y%m%d_%H%M%S")

pyroot=$ROOT/models/mmdet
export PYTHONPATH=${pyroot}:$PYTHONPATH

case $MODEL_NAME in
    "mask_rcnn_r50_caffe_fpn_mstrain_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "cascade_rcnn_r50_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "retinanet_r50_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/retinanet/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_r50_fpn_fp16_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "ssd300_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/ssd/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "faster_rcnn_r50_fpn_fp16_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_r50_caffe_fpn_mstrain_1x_coco.benchmark")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco.benchmark")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco.short")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "cascade_rcnn_r50_fpn_1x_coco.benchmark")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "retinanet_r50_fpn_1x_coco.benchmark")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/retinanet/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_r50_fpn_fp16_1x_coco.benchmark")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "ssd300_coco.benchmark")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/ssd/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "faster_rcnn_r50_fpn_fp16_1x_coco.benchmark")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_x101_32x4d_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "faster_rcnn_r50_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/faster_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "retinanet_r50_fpn_fp16_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_r101_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_x101_64x4d_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "fast_rcnn_r50_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/fast_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "cascade_mask_rcnn_r50_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "mask_rcnn_x101_64x4d_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/dcn/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    "fsaf_r50_fpn_1x_coco")
set -x

python -u models/mmdet/tools/train.py --config=configs/mmdet/fsaf/${MODEL_NAME}.py --launcher=mpi $EXTRA_ARGS \
2>&1 | tee $ROOT/log/mmdet/train.${MODEL_NAME}.log.$T
    ;;
    *)
      echo "invalid $MODEL_NAME"
      exit 1
      ;;
esac
