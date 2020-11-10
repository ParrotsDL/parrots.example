#!/bin/bash

mkdir -p log/SenseStar

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/SenseStar
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

work_path="models/SenseStar/sc2learner/experiments/alphastar_sl_baseline"

PYTHON_ARGS="python3 -u -m sc2learner.train.train_sl \
    --use_distributed \
    --config_path configs/SenseStar/config.yaml \
    --noonly_evaluate \
    --replay_list /mnt/lustre/share_data/parrots_model_data/sensewow/diff_size_total.train \
    --eval_replay_list /mnt/lustre/share_data/parrots_model_data/sensewow/diff_size_total.train"

srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=SenseStar_${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/SenseStar/train.${name}.log.$T

    