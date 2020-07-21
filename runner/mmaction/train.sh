#!/bin/bash
export PYTHONPATH=/mnt/lustre/jiaomenglei/parrots-dev/parrots2/python:$PYTHONPATH
export PYTHONPATH=~/dev/mmcv_debug/mmcv:$PYTHONPATH
mkdir -p log/mmaction

T=`date +%m%d%H%M`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/mmaction
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

case $name in
  "i3d_r50_video_32x2x1_100e_kinetics400_rgb")
    PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/recognition/i3d/${name}.py --launcher=slurm"
    ;;
  "tsn_r50_video_1x1x8_100e_kinetics400_rgb")
    PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/recognition/tsn/${name}.py --launcher=slurm"
    ;;
  "tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb")
    PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/recognition/tsn/${name}.py --launcher=slurm"
    ;;
  "i3d_r50_video_32x2x1_100e_kinetics400_rgb.benchmark")
    PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/recognition/i3d/${name}.py --launcher=slurm"
    ;;
  "tsn_r50_video_1x1x8_100e_kinetics400_rgb.benchmark")
    PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/recognition/tsn/${name}.py --launcher=slurm"
    ;;
  "tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb.benchmark")
    PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/recognition/tsn/${name}.py --launcher=slurm"
    ;;
  *)
    echo "invalid $name"
    exit 1
    ;; 
esac

set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=mmaction_${name} ${SRUN_ARGS}\
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmaction/train.${name}.log.$T

    