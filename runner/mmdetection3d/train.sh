mkdir -p log/mmdetection3d
now=$(date +"%Y%m%d_%H%M%S")
set -x

ROOT=.
pyroot=$ROOT/models/mmdetection3d
export PYTHONPATH=$pyroot:$PYTHONPATH

PARTITION=$1
GPUS=$2
MODEL=$3

case $MODEL in
    "votenet_16x8_sunrgbd-3d-10class")
MODEL_DIR="votenet"
;;
    "votenet_8x8_scannet-3d-18class")
MODEL_DIR="votenet"
;;
    "hv_second_secfpn_6x8_80e_kitti-3d-3class")
MODEL_DIR="second"
;;
    "hv_second_secfpn_6x8_80e_kitti-3d-car")
MODEL_DIR="second"
;;
    "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class")
MODEL_DIR="parta2"
;;
    "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car")
MODEL_DIR="parta2"
;;
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-car")
MODEL_DIR="pointpillars"
;; 
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class")
MODEL_DIR="pointpillars"
;;
    *)
echo "unknown model $MODEL"
exit 1
;;
esac

CONFIG=$ROOT/configs/mmdetection3d/${MODEL_DIR}/${MODEL}.py
WORK_DIR=$ROOT/log/mmdetection3d/${MODEL_DIR}/$MODEL

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

let COUNTER=0
while [ ! -d "./mmdet_mmdet3d_tmp" ];
do
    let COUNTER+=1
    if ((COUNTER > 3)); then
        break
    fi
    git clone -b v2.5.0 https://github.com/open-mmlab/mmdetection.git mmdet_mmdet3d_tmp
done
cd mmdet_mmdet3d_tmp
export PYTHONPATH=$(pwd):$PYTHONPATH
cd ..

srun -p ${PARTITION} \
     --job-name="mmdetection3d_${MODEL}" \
     --gres=gpu:8 \
     --ntasks=${GPUS} \
     --ntasks-per-node=8 \
     --cpus-per-task=5 \
     --kill-on-bad-exit=1 \
     ${SRUN_ARGS} \
     python -u $ROOT/models/mmdetection3d/tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${EXTRA_ARGS}

rm -rf mmdet_mmdet3d_tmp
