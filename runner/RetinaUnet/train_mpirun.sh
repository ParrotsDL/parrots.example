#!/bin/bash
set -x

source /usr/local/env/pat_latest

cd models/RetinaUnet
pip install -v -e . --user
cd ../../
## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

MODEL_NAME=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

mkdir -p log/RetinaUnet

T=`date +%m%d%H%M%S`
name=${MODEL_NAME}
ROOT=.

pyroot=$ROOT/models/RetinaUnet
export PYTHONPATH=$pyroot:$PYTHONPATH
export PYTHONPATH=$ROOT/configs/RetinaUnet/liver_ct_zssy_2d_runet:$PYTHONPATH

case $name in
    "2d_runet_infer")
      model_path=/mnt/lustre/share_data/jiaomenglei/model_pool_data/RetinaUnet/detection_codes/outputs/runet_2d_0703_nce+rce/fold_0/last_checkpoint
      image_save_path=$ROOT/log/RetinaUnet/infer_results
      mkdir -p ${image_save_path}
      PYTHON_ARGS="python -u models/RetinaUnet/exec_2d_runet.py --mode=test --exp_dir=${image_save_path}/logger \
                   --image_save_path=${image_save_path} --resume_to_checkpoint=${model_path} \
                   --exp_source=configs/RetinaUnet/liver_ct_zssy_2d_runet \
                   --modification='' "
      ;; 
    *)
      echo "invalid $name"
      exit 1
      ;; 
esac

set -x
$PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/RetinaUnet/train.liver_ct_zssy_${name}.log.$T