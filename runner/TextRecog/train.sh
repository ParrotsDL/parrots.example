#!/bin/bash

mkdir -p log/TextRecog/

T=`date +%m%d%H%M%S`
name=$3
ROOT=$(pwd)
cfg=$ROOT/configs/TextRecog/${name}.yaml
g=$(($2<8?$2:8))

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

pyroot=$ROOT/models/TextRecog
export PYTHONPATH=$pyroot:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}

cd models/TextRecog/
srun -p $1 --gres=gpu:1 -n 1 ${SRUN_ARGS} python setup.py bdist_wheel
srun -p $1 --gres=gpu:1 -n 1 ${SRUN_ARGS} pip install --user dist/text_recog-0.2.0-py3-none-any.whl
cd pytorch-ctc/pytorch-ctc-0.3.2
mkdir build
cd build
cmake ..
srun -p $1 --gres=gpu:1 -n 1 ${SRUN_ARGS} make
export WARP_CTC_PATH=`pwd`
cd ../parrots_binding
srun -p $1 --gres=gpu:1 -n 1 ${SRUN_ARGS} python setup.py install --user
cd $ROOT

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --cpus-per-task=3 --job-name=TextRecog_${name} ${SRUN_ARGS} \
python $ROOT/models/TextRecog/tools/train_val.py \
  --config=$cfg --phase train  ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/TextRecog/train.${name}.log.$T
