#!/bin/bash

mkdir -p log/3ddet/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3
BS=`expr 1 \* $2`
EPOCH=80
PY_ARGS=${@:4}
now=$(date +"%Y%m%d_%H%M%S")
ROOT=.

cfg=$ROOT/configs/3ddet/${JOB_NAME}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/3ddet
export LD_LIBRARY_PATH=/mnt/lustre/share_data/LOD/shishaoshuai/anaconda3/lib/python3.7/site-packages/spconv/:$LD_LIBRARY_PATH 
export PYTHONPATH=${pyroot}/:${pyroot}/parrots_spconv:${pyroot}/spconv:$PYTHONPATH

SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    ${SRUN_ARGS} --kill-on-bad-exit=1 \
python ${pyroot}/tools/train.py --cfg_file $cfg \
    --launcher slurm --batch_size $BS \
    --ckpt checkpoint/3ddet/${JOB_NAME}/ \
    --extra_tag bs_$BS --epochs $EPOCH ${PY_ARGS} \
    2>&1 | tee ${ROOT}/log/3ddet/train.${JOB_NAME}.log.${now}
