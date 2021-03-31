#!/bin/bash
set -x
mkdir -p log/ssd/
T=`date +%m%d%H%M%S`
name=$2
ROOT=.

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

source $1

export PYTHONPATH=$../../models/ssd/ssd.triondet/examples/coco-opencv/:$PYTHONPATH
export PYTHONPATH=$../../models/ssd/ssd.triondet/:$PYTHONPATH
export PYTHONPATH=$../../models/ssd/ssd.triondet/triondet/:$PYTHONPATH


# SRUN_ARGS=${SRUN_ARGS:-""}

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

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
$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/ssd/train.${name}.log.$T

