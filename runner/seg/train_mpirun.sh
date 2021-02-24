#!/bin/bash
set -x

source /usr/local/env/pat_latest

export PYTHONPATH=/senseparrots/python:/PAPExtension:/mnt/lustre/share/memcached:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/libmemcached/lib:$LD_LIBRARY_PATH
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

MODEL_NAME=$1
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

ROOT=.

mkdir -p log/Light_Seg/

EXP_DIR=checkpoints/
mkdir -p ${EXP_DIR}/result

now=$(date +"%Y%m%d_%H%M%S")

pyroot=$ROOT/models/Light_Seg

case $MODEL_NAME in
    "pspnet")

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
    python -u $pyroot/train.py \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    --config=configs/seg/pspnet.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${MODEL_NAME}.log.$T
    ;;
    "deeplab")
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
    python -u $pyroot/train.py \
    --config=configs/seg/deeplab.yaml $EXTRA_ARGS \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    2>&1 | tee $ROOT/log/seg/train.${MODEL_NAME}.log.$T
    ;;
    "mobilenet_v2_plus")
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
    python -u $pyroot/train.py --config=configs/seg/mobilenet_v2_plus.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${MODEL_NAME}.log.$T
    ;;
    "pspnet.benchmark")
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
    python -u $pyroot/train.py \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    --config=configs/seg/pspnet.benchmark.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${MODEL_NAME}.log.$T
    ;;
    "deeplab.benchmark")
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
    python -u $pyroot/train.py \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    --config=configs/seg/deeplab.benchmark.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${MODEL_NAME}.log.$T
    ;;
    "mobilenet_v2_plus.benchmark")
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
    python -u $pyroot/train.py --config=configs/seg/mobilenet_v2_plus.benchmark.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${MODEL_NAME}.log.$T
    ;;
    *)
      echo "invalid $MODEL_NAME"
      exit 1
      ;;
esac
