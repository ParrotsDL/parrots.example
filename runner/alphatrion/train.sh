#!/bin/bash

mkdir -p log/alphatrion/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/alphatrion/alphatrion
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

case $name in
    "mobilenet_v2_fp32")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/base_fp32.yaml \
      configs/alphatrion/mobilenet_v2.yaml  "
    ;;
    "mobilenet_v2_fp16")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/base_fp16.yaml \
      configs/alphatrion/mobilenet_v2.yaml  "
    ;;
    "se_resnet50_fp32")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/base_fp32.yaml \
      configs/alphatrion/se_resnet50.yaml  "
    ;;
    "se_resnet50_fp16")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/base_fp16.yaml \
      configs/alphatrion/se_resnet50.yaml  "
    ;;
    "resnet50_fp32")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/resnet50_fp32.yaml  "
    ;;
    "resnet50_fp16")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/resnet50_fp16.yaml  "
    ;;
    "resnet101_fp32")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/resnet101_fp32.yaml  "
    ;;
    "resnet101_fp16")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      -cfg configs/alphatrion/resnet101_fp16.yaml  "
    ;;
    "mobilenet_v2_fp32_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/base_fp32.yaml \
      configs/alphatrion/mobilenet_v2.yaml  "
    ;;
    "mobilenet_v2_fp16_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/base_fp16.yaml \
      configs/alphatrion/mobilenet_v2.yaml  "
    ;;
    "se_resnet50_fp32_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/base_fp32.yaml \
      configs/alphatrion/se_resnet50.yaml  "
    ;;
    "se_resnet50_fp16_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/base_fp16.yaml \
      configs/alphatrion/se_resnet50.yaml  "
    ;;
    "resnet50_fp32_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/resnet50_fp32.yaml  "
    ;;
    "resnet50_fp16_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/resnet50_fp16.yaml  "
    ;;
    "resnet101_fp32_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/resnet101_fp32.yaml  "
    ;;
    "resnet101_fp16_benchmark")
      PYTHON_ARGS="python -u models/alphatrion/alphatrion-example/bag_of_tricks/learn.py \
      --resume '' \
      --num_epochs 1 \
      -cfg configs/alphatrion/resnet101_fp16.yaml  "
    ;;
    *)
      echo "invalid $name"
      exit 1
      ;;
esac
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=alphatrion_${name} ${SRUN_ARGS}\
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/alphatrion/train.${name}.log.$T
                                                             
