#!/usr/bin/env bash

set -x

mkdir -p log/mmocr/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
cfg=$ROOT/configs/mmocr/${name}.py
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
PY_ARGS=${PY_ARGS:-"--validate"}
# PY_ARGS=${PY_ARGS:-""}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/mmocr/
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

git clone -b pat_v2.4.0 git@gitlab.sz.sensetime.com:parrotsDL-sz/mmdetection.git mmdet_mmocr
cd mmdet_mmocr
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

export PKG_CONFIG_PATH=/mnt/lustre/share/opencv-3.2/lib/pkgconfig
export LD_LIBRARY_PATH=/mnt/lustre/share/opencv-3.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/lustre/share/cuda-8.0-cudnnv6/lib64
# export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-10.0/lib64:$LD_LIBRARY_PATH

cd models/mmocr/mmocr/models/utils/instance_det_postprocessing
cp Makefile.pse Makefile && srun -p $1 --gres gpu:$g make
cp Makefile.pan Makefile && srun -p $1 --gres gpu:$g make
cd ../../../../../..

ln -s /mnt/lustre/share_data/parrots_model_data/mmocr/data data

srun --mpi=pmi2 -p $1 -n $2 --gres gpu:$g --ntasks-per-node $g --job-name=mmocr_${name} --kill-on-bad-exit=1 ${SRUN_ARGS} \
  python -u $ROOT/models/mmocr/tools/train.py \
  $cfg  --work-dir=work_dir/mmocr_${name} --launcher="slurm" ${PY_ARGS} ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/mmocr/train.${name}.log.$T

