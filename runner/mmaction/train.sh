#!/bin/bash

mkdir -p log/mmaction

T=`date +%m%d%H%M`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/mmaction
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHON_ARGS="python -u models/mmaction/tools/train.py configs/mmaction/${name}.py --launcher=slurm --validate"

set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=mmaction_${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmaction/train.${name}.log.$T

    