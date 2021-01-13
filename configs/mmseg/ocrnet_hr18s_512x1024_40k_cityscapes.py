_base_ = './ocrnet_hr18_512x1024_40k_cityscapes.py'
model = dict(
    pretrained='/mnt/lustre/share_data/parrots_model_ckpt/mmseg/mmseg_pretrain_model/hrnetv2_w18_small-b5a04e21.pth',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))
