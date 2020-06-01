#!/bin/sh
# test pape
#export PYTHONPATH=/mnt/lustre/zhuyuanhao/pape/refactor/PAPExtension/:$PYTHONPATH
p=$1
g=$(($2<8?$2:8))
# export NCCL_IB_HCA=mlx5_1
OMPI_MCA_mpi_warn_on_fork=0 srun -p $p --gres=gpu:$g -n $2 --ntasks-per-node=$g \
python -u benchmark_debug.py --benchmark --half --loss_scale 128.0 -p 100 --max_iter=4000 --bucket_size=4 -a alexnet_bn -b 128 /mnt/lustre/share/images
