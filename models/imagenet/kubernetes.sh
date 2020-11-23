#!/bin/bash

# 多机多卡的训练脚本
export PYTHONPATH=/senseparrots/python:/PAPExtension:/mnt/lustre/share/memcached:$PYTHONPATH
#export PYTHONPATH=/mnt/lustre/share/memcached:$PYTHONPATH
python3 -u main.py
#/mnt/lustre/xialei1/nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
