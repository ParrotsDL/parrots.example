#!/bin/bash
set -x
mkdir -p log/objectflow2_mot/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

now=$(date +"%Y%m%d_%H%M%S")
ROOT=${PWD}

pyroot=$ROOT/models/objectflow2_mot 
export PYTHONPATH=${pyroot}:$PYTHONPATH

# 编译mmcv
cd models/objectflow2_mot/
TORCHV_ERSION=`python -c "import torch; print(torch.__version__)" | tail -n 1`
if [ ${TORCHV_ERSION} == "parrots" ]
then
    git submodule update --init mmcv_pat
    ./build_mmcv_pat.sh $PARTITION
    export PYTHONPATH=$PWD:$PWD/mmdetection:$PWD/mmcv_pat:$PYTHONPATH
    export PARROTS_FORK_SAFE=ON
else
    git submodule update --init mmcv_pyt
    ./build_mmcv_pyt.sh $PARTITION
    export PYTHONPATH=$PWD:$PWD/mmdetection:$PWD/mmcv_pyt:$PYTHONPATH
fi
cd ${ROOT}

PARTITION=$1 \
TASK_CONFIG_LIST=\
configs/objectflow2_mot/face_1.py,\
configs/objectflow2_mot/face_2.py,\
configs/objectflow2_mot/body_2.py,\
configs/objectflow2_mot/body_3.py,\
configs/objectflow2_mot/body_5.py,\
configs/objectflow2_mot/hoi_1.py,\
configs/objectflow2_mot/mot_1.py,\
configs/objectflow2_mot/mot_2.py,\
configs/objectflow2_mot/mot_3.py,\
configs/objectflow2_mot/mot_4.py,\
configs/objectflow2_mot/mot_5.py,\
configs/objectflow2_mot/mot_6.py,\
configs/objectflow2_mot/mot_7.py,\
configs/objectflow2_mot/mot_8.py,\
configs/objectflow2_mot/mot_9.py,\
configs/objectflow2_mot/mot_10.py,\
configs/objectflow2_mot/mot_11.py,\
configs/objectflow2_mot/mot_12.py,\
configs/objectflow2_mot/mot_13.py \
TASK_QUOTA=\
5,3,4,3,1,8,1,1,1,1,4,1,1,1,1,1,1,1,1 \
GPUS=40 \
WORK_DIR=${WORK_DIR:-log/objectflow2_mot/$(date +"%Y%m%d_%H%M%S")/} \
GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:1}
srun \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ${pyroot}/tools/train_multitask.py ${TASK_CONFIG_LIST} ${TASK_QUOTA} --work-dir=${WORK_DIR} --launcher="slurm" --no-validate ${EXTRA_ARGS} \
    2>&1 | tee log/objectflow2_mot/train.log-$now

