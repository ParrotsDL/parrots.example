#/bin/sh
if [ ! -d "log/yolov3/" ]; then
  mkdir -p log/yolov3/
fi
T=`date +%m%d%H%M%S`
CUR_DIR=$(cd $(dirname "$pwd");pwd)

export PYTHONPATH=models/yolov3/:$PYTHONPATH

export LC_ALL=en_US.UTF-8
log_path=log/yolov3/
num_epochs=150
save_period=10
batch_size=128
data=models/yolov3/data/coco2017.yaml

# extra param of training
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
saved_path="yolov3_ckpt"

srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 \
--job-name=yolov3 python models/yolov3/tools/train.py \
--data ${data} --batch-size ${batch_size} --epochs ${num_epochs} \
--cfg models/yolov3/models/yolov3.yaml \
--hyp models/yolov3/data/hyps/hyp.scratch.yaml \
--save_period ${save_period} --noval --saved_path ${saved_path} \
${EXTRA_ARGS} 2>&1 | tee ${log_path}/yolov3_mlu290_train_${T}.log

# test model
last_epoch=$(( $num_epochs - 1 ))
test_batch_size=32
test_data=models/yolov3/data/coco2014.yaml
test_ckpt=${saved_path}/checkpoint_latest.pth

if [ -f "${test_ckpt}" ];then
    srun -p $1 -n1 --gres=mlu:1 --ntasks-per-node=1 \
    --job-name=yolov3_test python models/yolov3/tools/train.py \
    --data ${test_data} --batch-size ${test_batch_size} \
    --cfg models/yolov3/models/yolov3.yaml \
    --hyp models/yolov3/data/hyps/hyp.scratch.yaml \
    --noval --test --pretrained_model ${test_ckpt} \
    2>&1 | tee ${log_path}/yolov3_mlu290_test_${T}.log
else
    echo "Not found the test checkpoint: ${test_ckpt}, pls check!"
fi
