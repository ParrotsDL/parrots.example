#!/bin/bash
MODEL_NAME=resnet
NODE_NUM=1
GPU_NUM=8
CODE_DIR=/Users/wangqian4/Workspace/parrots.example/

NOW=$(date +"%Y%m%d-%H%M%S")
TOTAL_NUM=$((${NODE_NUM} * ${GPU_NUM}))
# INST_NAME=${MODEL_NAME,,}-${NODE_NUM}-pod-${TOTAL_NUM}-gpu-${NOW}
INST_NAME=${MODEL_NAME}-${NODE_NUM}-pod-${TOTAL_NUM}-gpu-${NOW}
export MODEL_NAME NODE_NUM GPU_NUM TOTAL_NUM INST_NAME CODE_DIR

envsubst < parrots_example_k8s.yaml | kubectl --kubeconfig ~/.kube/config-wangqian4 create -f -

export -n MODEL_NAME NODE_NUM GPU_NUM TOTAL_NUM INST_NAME CODE_DIR
