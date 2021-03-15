#!/bin/bash
source /usr/local/env/pat_latest

mkdir -p log/pod_v3.0
now=$(date +"%Y%m%d_%H%M%S")
set -x

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

ROOT=.
pyroot=$ROOT/models/pytorch-object-detection-v3.0/

name=$1
cfg=$ROOT/configs/pod_v3.0/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
g=$(($2<8?$2:8))
export PYTHONPATH=$pyroot:$PYTHONPATH

#srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g  \
#--job-name=pod_v3.0_${name} ${SRUN_ARGS} \
#mpirun -np $2 \
python -m pod train \
  --config=${cfg} \
  --display=1 \
   ${EXTRA_ARGS} \
  2>&1 | tee log/pod_v3.0/train_${name}.log-$now
