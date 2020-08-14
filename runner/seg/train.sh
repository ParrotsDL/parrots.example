#!/bin/bash

mkdir -p log/seg/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

pyroot=$ROOT/models/Light_Seg
export PYTHONPATH=$pyroot:$PYTHONPATH

#eval
EXP_DIR=checkpoints/
mkdir -p ${EXP_DIR}/result

case $name in
    "pspnet")
      #PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/pspnet.yaml"
      #PYTHON_ARGS1="python -u $pyroot/eval.py --config=configs/seg/pspnet.yaml"
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
      srun -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=seg_${name} ${SRUN_ARGS} \
    python -u $pyroot/train.py \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    --config=configs/seg/pspnet.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${name}.log.$T
    ;;
    "deeplab")
      #PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/deeplab.yaml"
      #PYTHON_ARGS1="python -u $pyroot/eval.py --config=configs/seg/deeplab.yaml"
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
      srun -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=seg_${name} ${SRUN_ARGS} \
    python -u $pyroot/train.py \
    --config=configs/seg/deeplab.yaml $EXTRA_ARGS \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    2>&1 | tee $ROOT/log/seg/train.${name}.log.$T
    ;;
    "mobilenet_v2_plus")
      #PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/mobilenet_v2_plus.yaml"
      #PYTHON_ARGS1=""
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
      srun -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=seg_${name} ${SRUN_ARGS} \
    python -u $pyroot/train.py --config=configs/seg/mobilenet_v2_plus.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${name}.log.$T
    ;;
    "pspnet.benchmark")
      #PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/pspnet.yaml"
      #PYTHON_ARGS1="python -u $pyroot/eval.py --config=configs/seg/pspnet.yaml"
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
      srun -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=seg_${name} ${SRUN_ARGS} \
    python -u $pyroot/train.py \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    --config=configs/seg/pspnet.benchmark.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${name}.log.$T
    ;;
    "deeplab.benchmark")
      #PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/deeplab.yaml"
      #PYTHON_ARGS1="python -u $pyroot/eval.py --config=configs/seg/deeplab.yaml"
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
      srun -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=seg_${name} ${SRUN_ARGS} \
    python -u $pyroot/train.py \
    --base_size=2048 \
    --scales 1.0 \
    --model_path=${EXP_DIR}best.pth \
    --save_folder=${EXP_DIR}result/ \
    --config=configs/seg/deeplab.benchmark.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${name}.log.$T
    ;;
    "mobilenet_v2_plus.benchmark")
      #PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/mobilenet_v2_plus.yaml"
      #PYTHON_ARGS1=""
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1\
      srun -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=seg_${name} ${SRUN_ARGS} \
    python -u $pyroot/train.py --config=configs/seg/mobilenet_v2_plus.benchmark.yaml $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${name}.log.$T
    ;;
    *)
      echo "invalid $name"
      exit 1
      ;;
esac
