srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 --job-name=yolov3 python train.py --data data/coco1984.yaml --batch-size 128 $3
#srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 --job-name=yolov3 python train.py --pretrained  yolov3_pretrained.pth $3
