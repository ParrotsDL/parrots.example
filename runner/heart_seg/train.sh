#!/bin/bash
mkdir -p log/heart_seg/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3
BS=`expr 4 \* $2`
EPOCH=3000

now=$(date +"%Y%m%d_%H%M%S")
ROOT=.

cfg=$ROOT/configs/heart_seg/${JOB_NAME}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/heart_seg
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH

SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=heart_seg_${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    ${SRUN_ARGS} --kill-on-bad-exit=1 \
python ${pyroot}/code/trainSegVNet.py --config ${cfg} \
    --fold 0 --lr 0.001 --batch-size ${BS} --num-workers ${BS} --base-features 16 \
    --loss cross_entropy --pre-load-data --val-interval 5 --test-interval 10\
    --output-dir "${pyroot}/OutSeg" --epochs ${EPOCH} ${EXTRA_ARGS} \
    2>&1 | tee ${ROOT}/log/heart_seg/train.${JOB_NAME}.log.${now}
