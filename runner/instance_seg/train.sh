#!/usr/bin/env bash
mkdir -p log/instance_seg/
part=$1
gpus=$2
ROOT=.
name=$3
# cfg=$ROOT/configs/instance_seg/${name}.yaml
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/instance_seg
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
 --mpi=pmi2 -p $part --gres=gpu:$2 -n$2 --ntasks-per-node=$2 --job-name=instance_${name} \
python models/instance_seg/train.py --config=$name \
 --dataset=coco2017_dataset --batch_size=8 \
 --save_folder='./models/instance_seg/weights/' \
${EXTRA_ARGS} 2>&1 | tee $ROOT/log/instance_seg/train.${name}.log.$T
