#!/bin/bash

mkdir -p log/mmdet

T=`date +%m%d%H%M`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/mmdet/tools
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}


case $name in
    "mask_rcnn_r50_caffe_fpn_mstrain_1x_coco")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "cascade_rcnn_r50_fpn_1x_coco")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "retinanet_r50_fpn_1x_coco")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/retinanet/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_fp16_1x_coco")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "ssd300_coco")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/ssd/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "faster_rcnn_r50_fpn_fp16_1x_coco")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_caffe_fpn_mstrain_1x_coco.benchmark")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco.benchmark")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_1x_coco.short")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/mask_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "cascade_rcnn_r50_fpn_1x_coco.benchmark")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/cascade_rcnn/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "retinanet_r50_fpn_1x_coco.benchmark")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/retinanet/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "mask_rcnn_r50_fpn_fp16_1x_coco.benchmark")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "ssd300_coco.benchmark")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/ssd/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    "faster_rcnn_r50_fpn_fp16_1x_coco.benchmark")
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 \
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=mmdet_${name} ${SRUN_ARGS}\
    python -u models/mmdet/tools/train.py --config=configs/mmdet/fp16/${name}.py --launcher=slurm $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmdet/train.${name}.log.$T
    ;;
    *)
      echo "invalid $name"
      exit 1
      ;;
esac
