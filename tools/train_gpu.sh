#/bin/sh
if [ ! -d "logs" ]; then
  mkdir logs
fi
T=`date +%m%d%H%M%S`
CUR_DIR=$(cd $(dirname "$pwd");pwd)

log_path=${CUR_DIR}/logs
num_epochs=150
save_period=10
batch_size=128
data=data/coco2017.yaml

# extra param of training
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 python tools/train.py \
--data ${data} --batch-size ${batch_size} \
--save_period ${save_period} --noval ${EXTRA_ARGS} \
2>&1 | tee ${log_path}/yolov3_gpu_${T}.log
