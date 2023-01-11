models_list=(
"mmcls resnet34_8xb32_in1k"
"mmcls vgg19_8xb32_in1k"
"mmcls vgg16bn_8xb32_in1k"
"mmcls resnet50_8xb32_in1k"
"mmcls resnetv1d50_8xb32_in1k"
"mmcls seresnet50_8xb32_in1k"
"mmcls mobilenet_v2_b32x8_imagenet"
"mmcls resnet101_8xb32_in1k"
"mmcls vgg16_8xb32_in1k"
"mmcls mobilenet_v3_large_imagenet"
"mmdet retinanet_r50_fpn_1x_coco"
"mmdet fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco"
"mmdet ssd300_voc0712"
"mmdet yolov3_d53_320_273e_coco"
"mmdet solo_r50_fpn_1x_coco"
)
models_num=${#models_list[@]}
max_parall=8

mkfifo ./fifo.$$& exec 798<> ./fifo.$$& rm -f ./fifo.$$
for ((i=0; i<$max_parall; i++)); do
    echo  "init add placed row $i" >&798
done

for ((i=0; i<$models_num; i++)); do
{
    read -u 798
    read frame model <<< ${models_list[i]}
    # sh ../smart/tools/cp_one_iter/runner/one_iter_run_half.sh ${SLURM_PAR_CAMB} none $frame $model
    echo $frame xxxxx $model
    sleep $i
    echo  "after add place row $i"  1>&798
}&
done

wait

echo Done

# The following models have precision problems:
# mmcls shufflenet-v2-1x_16xb64_in1k
# mmcls swin-base_16xb64_in1k
# mmdet centernet_resnet18_140e_coco
# mmdet faster_rcnn_r101_fpn_1x_coco
# mmdet mask_rcnn_r101_fpn_1x_coco
# mmdet autoassign_r50_fpn_8x2_1x_coco
# mmdet mask_rcnn_swin-t-p4-w7_fpn_1x_coco
# mmseg deeplabv3_r50-d8_512x1024_40k_cityscapes
# mmseg deeplabv3plus_r50-d8_512x1024_40k_cityscapes
# mmseg fcn_r50-d8_512x1024_40k_cityscapes
# mmseg deeplabv3_unet_s5-d16_64x64_40k_drive