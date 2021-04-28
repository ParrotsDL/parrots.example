#!/bin/bash
set -x
mkdir -p log/deformable_detr/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3
BS=`expr 4 \* $2`
EPOCH=400

now=$(date +"%Y%m%d_%H%M%S")
ROOT=${PWD}

cfg=$ROOT/configs/deformable_detr/${JOB_NAME}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/deformable_detr
export PYTHONPATH=${pyroot}/code/:$PYTHONPATH

EXP_DIR=exps/r50_deformable_detr
mkdir -p ${EXP_DIR}

# 编译算子
TORCHV_ERSION=`python -c "import torch; print(torch.__version__)" | tail -n 1`
if [ ${TORCHV_ERSION} != "parrots" ]
then
    cd models/deformable_detr/models/ops
    srun -p $1 --gres=gpu:1 -n1 python setup.py build install --user
    cd ${ROOT}
fi

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
        --with_box_refine \
        --two_stage \
        ${EXTRA_ARGS} \
        2>&1 | tee ${ROOT}/log/deformable_detr/train.${JOB_NAME}.log.${now}
