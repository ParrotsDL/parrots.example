#!/bin/bash
mkdir -p log/pod/

T=`date +%m%d%H%M`
ROOT=.
name=$3
cfg=$ROOT/configs/pod/${name}.yaml

pyroot=$ROOT/models/pytorch-object-detection
export PYTHONPATH=$pyroot:$PYTHONPATH

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --job-name=${name} \
python $ROOT/models/pytorch-object-detection/tools/train_val.py \
  --config=$cfg \
  2>&1 | tee log/pod/train.{name}.log.$T