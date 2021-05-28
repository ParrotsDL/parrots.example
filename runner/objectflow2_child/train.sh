#!/bin/bash
set -x
mkdir -p log/objectflow2_child/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

now=$(date +"%Y%m%d_%H%M%S")
ROOT=${PWD}

pyroot=$ROOT/models/objectflow2_child
export PYTHONPATH=${pyroot}:$PYTHONPATH

# 编译mmcv
cd models/objectflow2_child/
TORCHV_ERSION=`python -c "import torch; print(torch.__version__)" | tail -n 1`
if [ ${TORCHV_ERSION} == "parrots" ]
then
    ./build_mmcv_pat.sh $PARTITION
    export PYTHONPATH=$PWD:$PWD/mmdetection:$PWD/mmcv_pat:$PYTHONPATH
else
    ./build_mmcv_pyt.sh $PARTITION
    export PYTHONPATH=$PWD:$PWD/mmdetection:$PWD/mmcv:$PYTHONPATH
fi
cd ${ROOT}


TASK_CONFIG_LIST=${ROOT}/configs/objectflow2_child/1.py,${ROOT}/configs/objectflow2_child/1.1.py,${ROOT}/configs/objectflow2_child/1.2.py,${ROOT}/configs/objectflow2_child/2.py,${ROOT}/configs/objectflow2_child/2.1.py,${ROOT}/configs/objectflow2_child/2.2.py,${ROOT}/configs/objectflow2_child/3.py,${ROOT}/configs/objectflow2_child/4.py
TASK_QUOTA=2,1,1,2,1,1,4,4
WORK_DIR=log/objectflow2_child/$(date +"%Y%m%d_%H%M%S")


SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=objectflow2_child_${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    ${SRUN_ARGS} --kill-on-bad-exit=1 \
python -u ${pyroot}/tools/train_multitask.py ${TASK_CONFIG_LIST} ${TASK_QUOTA} --work-dir=${WORK_DIR} --launcher="slurm" ${EXTRA_ARGS} \
    2>&1 | tee log/objectflow2_child/train.log-$now
