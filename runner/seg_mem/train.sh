#!/usr/bin/env sh
mkdir -p log/Multi_organ_seg_HR
now=$(date +"%Y%m%d_%H%M%S")

ROOT=.
pyroot=$ROOT/models/Multi_organ_seg_HR
export PYTHONPATH=$pyroot:$PYTHONPATH

g=$(($2<8?$2:8))
name=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

cfg=$ROOT/configs/seg_mem/${name}.yaml
OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=seg_${name} -n$2 --gres=gpu:$g \
     --ntasks-per-node=$g   --kill-on-bad-exit=1 ${SRUN_ARGS} \
     python -u $pyroot/train.py  \
        --config ${cfg}  ${EXTRA_ARGS} \
        2>&1|tee log/Multi_organ_seg_HR/train_${name}.log-$now
