#!/bin/bash

mkdir -p log/Light_Nas_zpzhang/

T=`date +%m%d%H%M`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/Light_Nas_zpzhang
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

case $name in
    "single_path_oneshot")
      step=evolution
      PYTHON_ARGS="python -u models/Light_Nas_zpzhang/main.py \
         --config=configs/seg_nas/single_path_oneshot/config.yaml \
         --step=$step"
      ;; 
    "single_path_oneshot_benchmark")
      step=evolution
      PYTHON_ARGS="python -u models/Light_Nas_zpzhang/main.py \
         --config=configs/seg_nas/single_path_oneshot/config.benchmark.yaml \
         --step=$step"
      ;; 
    "single_path_oneshot_ceph_benchmark")
      step=evolution
      PYTHON_ARGS="python -u models/Light_Nas_zpzhang/main.py \
         --config=configs/seg_nas/single_path_oneshot/config_ceph.benchmark.yaml \
         --step=$step"
      ;;
    "single_path_oneshot_ceph")
      step=evolution
      PYTHON_ARGS="python -u models/Light_Nas_zpzhang/main.py \
         --config=configs/seg_nas/single_path_oneshot/config_ceph.yaml \
         --step=$step"
      ;;
    *)
      echo "invalid $name"
      exit 1
      ;; 
esac

set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=seg_nas_${name} ${SRUN_ARGS}\
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/Light_Nas_zpzhang/train.${name}.${step}.log.$T