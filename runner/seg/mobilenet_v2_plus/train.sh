#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../../../models/Light_Seg/
export PYTHONPATH=$ROOT:$PYTHONPATH
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n8 --gres=gpu:8 --ntasks-per-node=8 --job-name=py-rd2 --kill-on-bad-exit=1 \
python -u $ROOT/train.py --config=config.yaml  2>&1 | tee log_$now.txt
