#!/bin/bash
set -x

source $1

cd models/sr_v3.0/lib/_ext
pip install -v -e . --user
cd ../../../../

MODEL_NAME=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

mkdir -p log/sr_v3.0

T=`date +%m%d%H%M%S`
name=${MODEL_NAME}
ROOT=.

pyroot=$ROOT/models/sr_v3.0
export PYTHONPATH=$pyroot:$PYTHONPATH
export PYTHONPATH=$ROOT/models/sr_v3.0/lib:$PYTHONPATH

## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ $OMPI_COMM_WORLD_LOCAL_RANK == '0' ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi

NAME=`basename "$0"`
SIGN="x2_0_300_10f"

PYTHON_ARGS="python -m lib.train \
                --id                                                   		$NAME \
                --save_path                                        $ROOT/log/sr_v3.0/experiments \
                --configs                                               sr_v3.${name} \
                --isp_base_path    s3://parrots_model_data/SR_V3.0/xiaomi_F4/data/F4/isp_base.npy \
                --dataroot         s3://parrots_model_data/SR_V3.0/data/xiaomi_F4/F4/hdf5 \
                --motion_base_path s3://parrots_model_data/SR_V3.0/xiaomi_F4/data/motion_masks \
                --model                                              MRUSFT_ch4_b9 \
                --with_motion                                                    1 \
                --sharpen_reg                                                    1 \
                --merge_mode                                            bilinear \
                --scaling_factor                                               2.0 \
                --post_scaling                                                 1.0 \
                --patch_height                                                  64 \
                --patch_width                                                   64 \
                --offset_std                                                   2.0 \
                --align_noise_std                                             0.25 \
                --exposure_std                                                0.10 \
                --num_frames                                                    10 \
                --batch_size                                                    16 \
                --num_val                                                       50 \
                --workers                                                        8 \
                --learning_rate                                               1e-4 \
                --epochs                                                       400 \
                --lr_steps                                              60,120,240 \
                --lr_ratio                                                     0.1 \
                --summary_step                                                   5 \
                --save_step                                                      5 \
                --max_summ_img                                                   5"

set -x
$PYTHON_ARGS $EXTRA_ARGS 2>&1 | tee $ROOT/log/sr_v3.0/train.${name}.log.$T
