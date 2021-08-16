#!/bin/bash
set -x
mkdir -p log/objectflow2_object/
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
JOB_NAME=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

now=$(date +"%Y%m%d_%H%M%S")
ROOT=${PWD}

pyroot=$ROOT/models/objectflow2_object
export PYTHONPATH=${pyroot}:$PYTHONPATH

# 编译mmcv
cd models/objectflow2_object/
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

echo $CONDA_DEFAULT_ENV
CONDA_ROOT=/mnt/cache/share/platform/env/miniconda3.6
export MMCV_PATH=${CONDA_ROOT}/envs/${CONDA_DEFAULT_ENV}/mmcvs
mmcv_version=1.1.6
version=${mmcv_version#*=}
export PYTHONPATH=${MMCV_PATH}/$version:$PYTHONPATH

cd ${ROOT}


TASK_CONFIG_LIST=$ROOT/configs/objectflow2_object/1in2_retina_10ms_fpn_detcls_anchor3_agu.py,$ROOT/configs/objectflow2_object/2in2_retina_10ms_fpn_detreg_anchor3_agu.py
TASK_QUOTA=8,8
WORK_DIR=${WORK_DIR:-log/objectflow2_object/$(date +"%Y%m%d_%H%M%S")/}


SRUN_ARGS=${SRUN_ARGS:-""}
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${PARTITION} \
    --job-name=objectflow2_object_${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    ${SRUN_ARGS} --kill-on-bad-exit=1 \
python -u ${pyroot}/tools/train_multitask.py ${TASK_CONFIG_LIST} ${TASK_QUOTA} --work-dir=${WORK_DIR} --launcher="slurm" --no-validate ${EXTRA_ARGS} \
    2>&1 | tee log/objectflow2_object/train.log-$now
