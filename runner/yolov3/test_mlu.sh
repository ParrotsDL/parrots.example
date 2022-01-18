#/bin/sh
if [ ! -d "log/yolov3/" ]; then
  mkdir -p log/yolov3/
fi
T=`date +%m%d%H%M%S`
CUR_DIR=$(cd $(dirname "$pwd");pwd)

export PYTHONPATH=models/yolov3/:$PYTHONPATH
export LC_ALL=en_US.UTF-8

log_path=log/yolov3/
batch_size=32
data=models/yolov3/data/coco2014.yaml

# extra param of training
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 ${SRUN_ARGS} python models/yolov3/tools/train.py \
--cfg models/yolov3/models/yolov3.yaml \
--hyp models/yolov3/data/hyps/hyp.scratch.yaml \
--data ${data} --batch-size ${batch_size} \
--noval --test --pretrained_model $3 ${EXTRA_ARGS} \
2>&1 | tee ${log_path}/yolov3_mlu290_test_${T}.log
