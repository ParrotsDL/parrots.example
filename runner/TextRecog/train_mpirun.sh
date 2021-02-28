#!/bin/bash
set -x

source /usr/local/env/pat_latest

MODEL_NAME=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

if [ ! -d "models/TextRecog/pytorch-ctc/pytorch-ctc-0.3.2/build" ]; then
    sh ./setup.sh
else
    ext_path="models/TextRecog/pytorch-ctc/pytorch-ctc-0.3.2/parrots_binding/warp_ctc_pat.cpython-36m-x86_64-linux-gnu.so"
    while [ ! -f "${ext_path}" ];
    do
        sleep 5
    done
fi

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

mkdir -p log/TextRecog/

T=`date +%m%d%H%M%S`
name=${MODEL_NAME}
ROOT=.
cfg=$ROOT/configs/TextRecog/${name}_bn1.yaml

pyroot=$ROOT/models/TextRecog
export PYTHONPATH=$pyroot:$PYTHONPATH

if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cd models/TextRecog
    cd pytorch-ctc/pytorch-ctc-0.3.2
    mkdir build
    cd build
    cmake ..
    make
    export WARP_CTC_PATH=`pwd`
    cd ../parrots_binding
    pip install -v -e . --user
    cd ../../../../../
fi

export PYTHONPATH=models/TextRecog/pytorch-ctc/pytorch-ctc-0.3.2/parrots_binding:$PYTHONPATH

python -u models/TextRecog/tools/train_val.py \
  --config=$cfg --phase train --launcher mpi  ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/TextRecog/train.${name}.log.$T