#!/usr/bin/env bash

set -x

mkdir -p log/mmocr_ctc/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
cfg=$ROOT/configs/mmocr_ctc/${name}.py
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
PY_ARGS=${PY_ARGS:-"--validate"}
# PY_ARGS=${PY_ARGS:-""}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/mmocr_ctc/
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

git clone -b pat_v2.4.0 git@gitlab.sz.sensetime.com:parrotsDL-sz/mmdetection.git mmdet_mmocr
cd mmdet_mmocr
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g --job-name=mmocr_ctc_${name} --kill-on-bad-exit=1 ${SRUN_ARGS} \
  python -u $ROOT/models/mmocr_ctc/tools/train.py \
  $cfg  --work-dir=$ROOT/models/mmocr_ctc/work_dir/${name} --launcher="slurm" ${PY_ARGS} ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/mmocr_ctc/train.${name}.log.$T
