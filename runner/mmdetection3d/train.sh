mkdir -p log/mmdetection3d
now=$(date +"%Y%m%d_%H%M%S")
set -x

ROOT=.
pyroot=$ROOT/models/mmdetection3d
export PYTHONPATH=$pyroot:$PYTHONPATH

PARTITION=$1
GPUS=$2
MODEL=$3
CONFIG=$ROOT/configs/mmdetection3d/${MODEL}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
     --job-name="mmdetection3d_${MODEL}" \
     --gres=gpu:8 \
     --ntasks=${GPUS} \
     --ntasks-per-node=8 \
     --cpus-per-task=5 \
     --kill-on-bad-exit=1 \
     ${SRUN_ARGS} \
     python -u tools/train.py ${CONFIG} --word-dir=./ --launcher="slurm" ${EXTRA_ARGS}
