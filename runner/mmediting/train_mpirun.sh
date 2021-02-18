#!/bin/bash
set -x

source /usr/local/env/pat_latest 


MODEL_NAME=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

mkdir -p log/mmediting

T=`date +%m%d%H%M%S`

ROOT=.
pyroot=$ROOT/models/mmediting
export PYTHONPATH=$pyroot:$PYTHONPATH
export PARROTS_POOL_DATALOADER=1
export PYTHONPATH=/mnt/lustre/share/memcached:$PYTHONPATH



## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi



PYTHON_ARGS="python -u models/mmediting/tools/train.py configs/mmediting/${MODEL_NAME}.py --launcher=mpi"

set -x
#OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
#srun -K --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=mmediting_${name} ${SRUN_ARGS}\
#mpirun -np $2 \
$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmediting/train.${MODEL_NAME}.log.$T

    