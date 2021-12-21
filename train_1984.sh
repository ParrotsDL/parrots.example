#!/bin/sh
T=`date +%m%d%H%M%S`
srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 python train.py --data data/coco1984.yaml \
--save_period=10 --batch-size 128 --noval 2>&1 | tee yolov3_1984_${T}.log
#srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 --job-name=yolov3 \
#python train.py --pretrained  yolov3_pretrained.pth --saved_path=./saved_ckpt/
