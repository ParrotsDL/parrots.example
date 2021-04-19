#!/bin/bash

mkdir -p log/vision_transformer
now=$(date +"%Y%m%d_%H%M%S")
# set -x
ROOT=.
pyroot=$ROOT/models/vision_transformer
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
# g=$(($2<8?$2:8))
g=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

str=`python  -c "import torch;print(torch.__version__)"`
flag=$(echo $str | grep "parrots")
if [ -n "$flag" ]; then
    env="pa"
else
    env="pt"
fi

python $ROOT/models/vision_transformer/run.py --train -${env} --partition $1 --gpus $g \
--arch ${name} \
-j vision_transformer_${name} \
--script \
${EXTRA_ARGS} \
2>&1 | tee log/vision_transformer/train_${name}.log-$now      
