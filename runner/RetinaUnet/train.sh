#!/bin/bash

mkdir -p log/RetinaUnet

T=`date +%m%d%H%M%S`
name=$3
ROOT=.
EXTRA_ARGS=${@:4}

pyroot=$ROOT/models/RetinaUnet
export PYTHONPATH=$pyroot:$PYTHONPATH
export PYTHONPATH=$ROOT/configs/RetinaUnet/liver_ct_zssy_2d_runet:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

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
srun -p $1 -n$2 --gres gpu:$g --ntasks-per-node $g --job-name=RetinaUnet_${name} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/RetinaUnet/train.liver_ct_zssy_${name}.log.$T