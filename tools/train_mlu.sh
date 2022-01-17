#/bin/sh
set -e

if [ ! -d "logs" ]; then
  mkdir logs
fi
T=`date +%m%d%H%M%S`
CUR_DIR=$(cd $(dirname "$pwd");pwd)

log_path=${CUR_DIR}/logs
num_epochs=150
save_period=10
batch_size=128
saved_path="yolov3_ckpt"  # path for save checkpoint
train_data=data/coco2017.yaml

# extra param of training
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

# train model
srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 ${SRUN_ARGS} python tools/train.py \
--data ${train_data} --batch-size ${batch_size} --epochs ${num_epochs} \
--save_period ${save_period} --noval --saved_path ${saved_path} ${EXTRA_ARGS} \
2>&1 | tee ${log_path}/yolov3_mlu290_${T}.log

# test model
last_epoch=$(( $num_epochs - 1 ))
test_batch_size=$(( $batch_size / $2 ))
test_data=data/coco2014.yaml
test_ckpt=${saved_path}/checkpoint_latest.pth

if [ -f "${test_ckpt}" ]; then
    srun -p $1 -n1 --gres=mlu:1 --ntasks-per-node=1 \
    --job-name=yolov3 ${SRUN_ARGS} python tools/train.py \
    --data ${test_data} --batch-size ${test_batch_size} \
    --noval --test --pretrained_model ${test_ckpt} \
    2>&1 | tee ${log_path}/yolov3_mlu290_test_${T}.log
else
    echo "Not found the latest checkpoint: ${test_ckpt}, please check!"
fi
