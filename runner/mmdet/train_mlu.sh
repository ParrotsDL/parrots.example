#!/bin/bash

mkdir -p log/mmdet
export PYTORCH_VERSION=1.4

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/mmdet
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}


# 避免mm系列重复打印
export PARROTS_DEFAULT_LOGGER=FALSE

case $name in
    "mask_rcnn_r50_fpn_mstrain-poly_3x_coco.benchmark")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "cascade_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "retinanet_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/retinanet/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_fp16_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "ssd300_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/ssd/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "faster_rcnn_r50_fpn_fp16_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "faster_rcnn_r50_fpn_mstrain_3x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/faster_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_mstrain-poly_3x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco.short")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "cascade_rcnn_r50_fpn_1x_coco.benchmark")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "retinanet_r50_fpn_1x_coco.benchmark")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/retinanet/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_fp16_1x_coco.benchmark")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "ssd300_coco.benchmark")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/ssd/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "faster_rcnn_r50_fpn_fp16_1x_coco.benchmark")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_x101_32x4d_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "faster_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/faster_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "retinanet_r50_fpn_fp16_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r101_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_x101_64x4d_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "fast_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fast_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "cascade_mask_rcnn_r50_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_x101_64x4d_fpn_1x_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "yolov3_d53_mstrain-416_273e_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/yolo/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    " yolov3_d53_mstrain-608_273e_coco")
set -x

srun -p $1 -n$2 \
        --gres mlu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/yolo/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    *)
      echo "invalid $name"
      exit 1
      ;;
esac