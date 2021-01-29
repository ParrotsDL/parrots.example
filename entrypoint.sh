NAMESPACE="wenli"

JOB_NAME="${NAMESPACE}-test"

export container_job_name=${JOB_NAME}

spc cancel mpi-job -j "${JOB_NAME}" -N ${NAMESPACE}
spc cancel pod -p "${JOB_NAME}-1" -N ${NAMESPACE}
spc cancel pod -p "${JOB_NAME}-2" -N ${NAMESPACE}

sh runner/mmediting/spc_run.sh wenli  mmediting  cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter 1  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500

spc log -N ${NAMESPACE} -p ${JOB_NAME}-1 --sync

#cyclegan_lsgan
#edsr_x2c64b16_g1_300k_div2k
#dim_stage2_v16_pln_1x1_1000k_comp1k
#gl_256x256_8x12_celeba

#
#srcnn_x4k915_g1_1000k_div2k

# done PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 srcnn_x4k915_g1_1000k_div2k --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 8 deepfillv2_256x256_8x2_celeba  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
#PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 dim_stage2_v16_pln_1x1_1000k_comp1k  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 edsr_x2c64b16_g1_300k_div2k  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 esrgan_psnr_x4c64b23g32_g1_1000k_div2k  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 gl_256x256_8x12_celeba  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
#PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 indexnet_mobv2_1x16_78k_comp1k  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 msrresnet_x4c64b16_g1_1000k_div2k  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 pconv_256x256_stage1_8x1_celeba  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500
# PARROTS_BENCHMARK=1 sh runner/mmediting/train.sh pat_dev 1 pix2pix_vanilla_unet_bn_1x1_80k_facades  --pavi --pavi-project parrots_test --data-reader CephReader --seed 1024 --max-step 500


