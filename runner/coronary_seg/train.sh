#!/bin/bash
mkdir -p log/coronary_seg/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3
BS=`expr 2 \* $2`
EPOCH=500

now=$(date +"%Y%m%d_%H%M%S")
ROOT=.

cfg=$ROOT/configs/coronary_seg/${JOB_NAME}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/coronary_seg
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH

SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=coronary_seg_${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --ntasks-per-node=1 \
    ${SRUN_ARGS} --kill-on-bad-exit=1 \
python ${pyroot}/code/trainSegVNet.py --config ${cfg} \
    --fold 0 --lr 0.001 --batch-size ${BS} --num-workers ${BS} --base-features 16 \
    --loss 'mix' --resample 1.0 --loss-balance 0.3 --save-model-interval 50  \
    --output-dir ${pyroot}/OutSegTrain\
    --epochs ${EPOCH}  ${EXTRA_ARGS} \
    2>&1 | tee ${ROOT}/log/coronary_seg/train.${JOB_NAME}.log.${now}