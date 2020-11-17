#!/bin/bash
mkdir -p log/deformable_detr/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3
BS=`expr 4 \* $2`
EPOCH=400

now=$(date +"%Y%m%d_%H%M%S")
ROOT=.

cfg=$ROOT/configs/deformable_detr/${JOB_NAME}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/deformable_detr
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH

EXP_DIR=exps/r50_deformable_detr
mkdir -p ${EXP_DIR}

SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=deformable_detr_${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    ${SRUN_ARGS} --kill-on-bad-exit=1 \
python  -u ${pyroot}/main.py \
        --config ${cfg} \
        --batch_size 2 \
        --num_workers 4 \
        --cache_mode \
        --output_dir ${EXP_DIR} \
        --checkpoint_enc_ffn \
        --checkpoint_dec_ffn \
        ${EXTRA_ARGS} \
        2>&1 | tee ${ROOT}/log/deformable_detr/train.${JOB_NAME}.log.${now}
