#!/bin/bash
mkdir -p log/weekly/pod/

T=`date +%m%d%H%M`
ROOT=~/parrots.test
cfg=$ROOT/configs/weekly/pod/rfcn-R101-ohem-deform-1x.yaml

pyroot=$ROOT/models/pytorch-object-detection
export PYTHONPATH=$pyroot:$PYTHONPATH

srun --mpi=pmi2 -p $1 -n16 --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=R50-C4 \
python $ROOT/models/pytorch-object-detection/tools/train_val.py \
  --config=$cfg \
  2>&1 | tee log/weekly/pod/train.rfcn-R101-ohem-deform-1x.log_latest.$T
