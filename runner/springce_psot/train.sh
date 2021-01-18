#!/bin/bash

mkdir -p log/springce_psot/

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
test_dataset=VOT2018

cfg=$PWD/configs/springce_psot/${name%%_*}/${name#*_}/config.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/springce_psot
if [ ! -d "${pyroot}/testing_dataset/${test_dataset}" ]; then
  ln -s /s3://parrots_model_data/PSOT/ucg.tracking.academic.test/${test_dataset} ${pyroot}/testing_dataset/${test_dataset}
fi

export PYTHONPATH=$pyroot/experiments/${name%%_*}/${name#*_}:$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

cd $pyroot
python setup.py build_ext --inplace
cd -

srun --mpi=pmi2 -p $1 -n`expr 2 \* $2` --gres=gpu:$g --ntasks-per-node=$g --job-name=springce_psot_${name} ${SRUN_ARGS} \
  python -u -m siamrpn train \
  --cfg=$cfg --test_dataset=${test_dataset} ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/springce_psot/train.${name}.log.$T