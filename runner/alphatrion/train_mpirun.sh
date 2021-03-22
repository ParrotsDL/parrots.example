#!/bin/bash
source $1
mkdir -p log/alphatrion/

T=`date +%m%d%H%M%S`
name=$2
ROOT=.

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

pyroot=$ROOT/models/alphatrion/alphatrion
export PYTHONPATH=$pyroot:$PYTHONPATH

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
$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/alphatrion/train.${name}.log.$T
                                                             
