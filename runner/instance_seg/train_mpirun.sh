#!/usr/bin/env bash
source pat_latest
mkdir -p log/instance_seg/
ROOT=.
name=$1
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

pyroot=$ROOT/models/instance_seg
export PYTHONPATH=$pyroot:$PYTHONPATH

python models/instance_seg/train.py --config=$name \
 --dataset=coco2017_dataset --batch_size=8 \
 --save_folder='./models/instance_seg/weights/' \
${EXTRA_ARGS} 2>&1 | tee $ROOT/log/instance_seg/train.${name}.log.$T
