#/bin/sh
T=`date +%m%d%H%M%S`

srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 python train.py \
--data data/coco2014.yaml --batch-size=64 \
--save_period 10 --noval 2>&1 | tee  yolov3_mlu290_${T}.log
#srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 --job-name=yolov3 \
#python train.py --pretrained  yolov3_pretrained.pth
