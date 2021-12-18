srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 --job-name=yolov3 python train.py --data data/coco2014.yaml --save-period 30 $3
#srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 --job-name=yolov3 python train.py --pretrained  yolov3_pretrained.pth $3
