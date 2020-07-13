#!/bin/bash

mkdir -p log/seg/

T=`date +%m%d%H%M`
name=$3
ROOT=.

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

pyroot=$ROOT/models/Light_Seg
export PYTHONPATH=$pyroot:$PYTHONPATH

case $name in
    "pspnet")
      PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/pspnet.yaml"
    ;;
    "deeplab")
      PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/deeplab.yaml"
    ;;
    "mobilenet_v2_plus")
      PYTHON_ARGS="python -u $pyroot/train.py --config=configs/seg/mobilenet_v2_plus.yaml"
    ;;
    *)
      echo "invalid $name"
      exit 1
      ;;
esac
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=seg_${name} ${SRUN_ARGS}\
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/seg/train.${name}.log.$T
                                                             
