#!/bin/sh
mkdir -p engine_prof_log
now=$(date +"%Y%m%d_%H%M%S")

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH

g=$(($2<8?$2:8))

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=engine_prof \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g \
    python -u ../tools/engine_prof.py 2>&1 | tee engine_prof_log/train_engine.log-$now
