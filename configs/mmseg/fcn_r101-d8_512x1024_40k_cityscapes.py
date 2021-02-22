_base_ = './fcn_r50-d8_512x1024_40k_cityscapes.py'
model = dict(pretrained='/mnt/lustre/share_data/parrots_model_ckpt/mmseg/mmseg_pretrain_model/resnet101c128_csail-159f67a3.pth', backbone=dict(depth=101))
