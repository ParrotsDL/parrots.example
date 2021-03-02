#!/bin/bash
set -x

source pat_latest

# 多机多卡的训练脚本
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

MODEL_NAME=$1
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

ROOT=.

mkdir -p log/pattern_v2_5_sp

T=`date +%m%d%H%M%S`

cfg=$ROOT/configs/pattern_v2_5_sp/${MODEL_NAME}.yaml

pyroot=$ROOT/models/pattern_v2_5_sp
export PYTHONPATH=$pyroot:$PYTHONPATH

python $ROOT/models/pattern_v2_5_sp/tools/dist_train.py \
  --config=$cfg ${EXTRA_ARGS} \
  --now $now $T \
  2>&1 | tee $ROOT/log/pattern_v2_5_sp/train.${MODEL_NAME}.log.$T

# model_folder=$(cat $cfg | shyaml get-value strategy.save_path)
if [ -z $PARROTS_BENCHMARK ]; then
    test_iters=200000
    model_name="iter_${test_iters}_ckpt.pth.tar"
    echo "testing: "${model_folder}$model_name

    python $ROOT/models/pattern_v2_5_sp/tools/dist_test.py \
      --config=$cfg ${EXTRA_ARGS} \
      -e --model-name=$model_name \
      2>&1 | tee $ROOT/log/pattern_v2_5_sp/test.${MODEL_NAME}.log.$T
fi