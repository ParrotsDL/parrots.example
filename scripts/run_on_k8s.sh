#!/bin/sh
MODEL_NAME=resnet
NODE_NUM=1
GPU_NUM=8
MAX_EPOCH=120
CODE_DIR=/mnt/lustrenew/wangqian4/parrots.example

NOW=$(date +"%Y%m%d-%H%M%S")
TOTAL_NUM=$((${NODE_NUM} * ${GPU_NUM}))
INST_NAME=${MODEL_NAME,,}-${NODE_NUM}-pod-${TOTAL_NUM}-gpu-${MAX_EPOCH}-epoch-${NOW}
export MODEL_NAME NODE_NUM GPU_NUM MAX_EPOCH TOTAL_NUM INST_NAME CODE_DIR

envsubst < parrots_example_k8s.yaml | kubectl create -f -

export -n MODEL_NAME NODE_NUM GPU_NUM MAX_EPOCH TOTAL_NUM INST_NAME CODE_DIR
