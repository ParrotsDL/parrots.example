#!/usr/bin/env bash
mkdir -p log/Pattern
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/Pattern
export PYTHONPATH=$pyroot:$PYTHONPATH



array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}


partition=$1
train_gpu=$2
config=$3


SRUN_ARGS=${SRUN_ARGS}  sh runner/Pattern/train_1.sh $partition $train_gpu   $config  ${EXTRA_ARGS} 
SRUN_ARGS=${SRUN_ARGS} sh runner/Pattern/test_1.sh  $partition 1  $config   ${EXTRA_ARGS} 

