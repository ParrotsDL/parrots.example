#!/bin/bash

mkdir -p log/mmseg/
cd models/mmsegmentation/
pip install -e.
cd ../../
T=`date +%m%d%H%M%S`
name=$3
ROOT=.
cfg=$ROOT/configs/mmseg/${name}.py
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/mmsegmentation
export PYTHONPATH=$pyroot:$PYTHONPATH
export PARROTS_ALIGN_TORCH=1
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=mmseg_${name} --cpus-per-task=5 --kill-on-bad-exit=1 ${SRUN_ARGS} \
python $ROOT/models/mmsegmentation/tools/train.py \
  $cfg --launcher="slurm" ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/mmseg/train.${name}.log.$T
