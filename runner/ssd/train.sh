#!/bin/bash

mkdir -p log/ssd/
T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

export PYTHONPATH=$../../models/ssd/ssd.triondet/examples/coco-opencv/:$PYTHONPATH
export PYTHONPATH=$../../models/ssd/ssd.triondet/:$PYTHONPATH
export PYTHONPATH=$../../models/ssd/ssd.triondet/triondet/:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}


########Arguments#########
case $name in
    "ssd_FSAF")
      PYTHON_ARGS="python -u models/ssd/ssd.triondet/examples/tools/train.py \
      -cfg configs/ssd/coco896-FSAF-Res50.yaml"
    ;;
    "ssd_Retina")
      PYTHON_ARGS="python -u models/ssd/ssd.triondet/examples/tools/train.py \
      -cfg configs/ssd/coco896-Retina-R50-ce-e14.yaml"  
    ;;
    "ssd_FSAF_benchmark")
      PYTHON_ARGS="python -u models/ssd/ssd.triondet/examples/tools/train.py \
      -cfg configs/ssd/coco896-FSAF-Res50.benchmark.yaml"
    ;;
    "ssd_Retina_benchmark")
      PYTHON_ARGS="python -u models/ssd/ssd.triondet/examples/tools/train.py \
      -cfg configs/ssd/coco896-Retina-R50-ce-e14.benchmark.yaml"
    ;;
    *)
      echo "invalid $name"
      exit 1
      ;;
esac
set -x
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K -p $1 -n$2  --gres gpu:8 --ntasks-per-node 8 --job-name=${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/ssd/train.${name}.log.$T

