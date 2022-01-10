#/bin/sh
if [ ! -d "log/yolov3/" ]; then
  mkdir -p log/yolov3/
fi
T=`date +%m%d%H%M%S`
CUR_DIR=$(cd $(dirname "$pwd");pwd)

export PYTHONPATH=models/yolov3/:$PYTHONPATH

log_path=log/yolov3/
num_epochs=150
save_period=10
batch_size=128
data=models/yolov3/data/coco2017.yaml

# extra param of training
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 python models/yolov3/tools/train.py \
--data ${data} --batch-size ${batch_size} \
--cfg models/yolov3/models/yolov3.yaml \
--hyp models/yolov3/data/hyps/hyp.scratch.yaml \
--save_period ${save_period} --noval ${EXTRA_ARGS} \
2>&1 | tee ${log_path}/yolov3_mlu290_${T}.log
