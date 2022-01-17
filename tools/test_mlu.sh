#/bin/sh
if [ ! -d "logs" ]; then
  mkdir logs
fi
T=`date +%m%d%H%M%S`
CUR_DIR=$(cd $(dirname "$pwd");pwd)

log_path=${CUR_DIR}/logs
batch_size=32
data=data/coco2014.yaml

# extra param of training
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 ${SRUN_ARGS} python tools/train.py \
--data ${data} --batch-size ${batch_size} \
--noval --test --pretrained_model $3 ${EXTRA_ARGS} \
2>&1 | tee ${log_path}/yolov3_mlu290_test_${T}.log
