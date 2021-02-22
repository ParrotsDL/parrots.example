#!/bin/bash
set -x
source /usr/local/env/pat_latest

name=$1
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
 
## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

mkdir -p log/light_nas/
now=$(date +"%Y%m%d_%H%M%S")
ROOT=.
pyroot=$ROOT/models/light_nas
export PYTHONPATH=$pyroot:$PYTHONPATH
cfg=$ROOT/configs/light_nas/${name}.yaml
step='search'

case $name in
    "single_path_oneshot_search")
      step=search
      PYTHON_ARGS="python -m main \
         --config=configs/light_nas/single_path_oneshot/search.yaml \
         --step=$step"
      ;; 
    "single_path_oneshot_evolution")
      step=evolution
      PYTHON_ARGS="python -m main \
         --config=configs/light_nas/single_path_oneshot/evolution.yaml \
         --step=$step"
      ;; 
    *)
      echo "invalid $name"
      exit 1
      ;; 
esac

$PYTHON_ARGS ${EXTRA_ARGS} \
    2>&1 | tee log/light_nas/train_${step}_${name}.log-$now
