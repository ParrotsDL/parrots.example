#!/usr/bin/env sh
mkdir -p log/mmtrack
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/mmtrack
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$1
cfg=$ROOT/configs/mmtrack/${name}.py
work_dir=$pyroot/work_dirs/${name}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

best_epoch=50
p1=0.5
p2=0.05
p3=0.6

if [[ $3 == "siamrpnpp" ]]; then
    python $pyroot/mmtrack/face_eval_plus/quantitative/meval.py -c ${cfg} -m SiamRPN -b 8 -e 30 -r ${work_dir}/result-newpart/ -w ${work_dir}/ -rf 20 -bf 1 \
    -v /mnt/lustre/share_data/hanyachao/mmtrack/jsons/pm_v2.json \
    /mnt/lustre/share_data/hanyachao/mmtrack/test_data/nme_test/easycase/facial_landmark_nme_easycase.json\
    /mnt/lustre/share_data/hanyachao/mmtrack/jsons/facial_landmark_video_large_yaw_latest_0604_test.json \
    /mnt/lustre/share_data/hanyachao/mmtrack/jsons/occ_videos/occ_wangchao0611.json \
    -d s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/testset/test_video_from_pm_v2/ \
    s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/testset/face_nme_test/easycase/ \
    s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/large_angle_data/large_yaw/ \
    /mnt/lustre/share_data/hanyachao/mmtrack/test_data/occlusion_test/useful_video/testset_ours/json_label/final_image/ \
    ${EXTRA_ARGS} \
    | tee ${work_dir}/eval-$now.log
else
    python $pyroot/mmtrack/face_eval_plus/quantitative/meval.py -c ${cfg} -m SiamRPN -b 8 -e 50 -r ${work_dir}/result-newpart/ -w ${work_dir}/ -rf 20 -bf 1 \
    -v /mnt/lustre/share_data/hanyachao/mmtrack/jsons/pm_v2.json \
    /mnt/lustre/share_data/hanyachao/mmtrack/test_data/nme_test/easycase/facial_landmark_nme_easycase.json\
    /mnt/lustre/share_data/hanyachao/mmtrack/jsons/facial_landmark_video_large_yaw_latest_0604_test.json \
    /mnt/lustre/share_data/hanyachao/mmtrack/jsons/occ_videos/occ_wangchao0611.json \
    -d s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/testset/test_video_from_pm_v2/ \
    s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/testset/face_nme_test/easycase/ \
    s3://parrots_model_data/mmtrack/ARFace.Head_SOT_bucket/trainset/large_angle_data/large_yaw/ \
    /mnt/lustre/share_data/hanyachao/mmtrack/test_data/occlusion_test/useful_video/testset_ours/json_label/final_image/ \
    ${EXTRA_ARGS} \
    | tee ${work_dir}/eval-$now.log
fi
