#!/bin/sh
EXP_DIR=checkpoints/
mkdir -p ${EXP_DIR}/result
now=$(date +"%Y%m%d_%H%M%S")
ROOT=../../../models/Light_Seg
export PYTHONPATH=$ROOT:$PYTHONPATH

srun -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=pred --kill-on-bad-exit=1 \
python $ROOT/eval.py \
  --base_size=2048 \
  --scales 1.0 \
  --config=config.yaml \
  --model_path=${EXP_DIR}best.pth \
  --save_folder=${EXP_DIR}result/ \
  2>&1 | tee ${EXP_DIR}/result/eva-$now.log
