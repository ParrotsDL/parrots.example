#!/usr/bin/env bash
mkdir -p log/mmtrack
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/mmtrack
export PYTHONPATH=$pyroot:$PYTHONPATH

p=$1
name=$3
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

SRUN_ARGS=${SRUN_ARGS} sh runner/mmtrack/run_train.sh $p $g $name ${EXTRA_ARGS}
SRUN_ARGS=${SRUN_ARGS} sh runner/mmtrack/run_test.sh $name ${EXTRA_ARGS} 
