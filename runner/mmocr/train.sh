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
# PY_ARGS=${PY_ARGS:-"--validate"}
PY_ARGS=${PY_ARGS:-""}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/mmocr/mmocr
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

ln -s /mnt/lustre/share_data/parrots_model_data/mmocr/data data
srun --mpi=pmi2 -p $1 -n $2 --gres gpu:$g --ntasks-per-node $g --job-name=mmocr_${name} --kill-on-bad-exit=1 ${SRUN_ARGS} \
  python -u $ROOT/models/mmocr/tools/train.py \
  $cfg  --work-dir=work_dir/mmocr_${name} --launcher="slurm" ${PY_ARGS} ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/mmocr/train.${name}.log.$T

# pip uninstall mmdet