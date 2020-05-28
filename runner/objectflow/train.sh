#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
partition=$1
job_name=$2
gpus=${3:-8}
gpu_per_node=8
config=configs/objectflow/${job_name}.py

work_dir=log/objectflow/$2
mkdir -p ${work_dir}

ROOT=.
pyroot=$ROOT/models/ObjectFlow

export PYTHONPATH=$pyroot:$PYTHONPATH
srun -p ${partition} --gres=gpu:${gpu_per_node} -n${gpus} \
    --ntasks-per-node=${gpu_per_node} \
    --job-name=${job_name} --kill-on-bad-exit=1 \
python -u ${pyroot}/tools/train.py ${config} --work_dir=${work_dir} \
    --launcher='slurm' --validate \
    2>&1 | tee ${work_dir}/train_${job_name}.log-$now
