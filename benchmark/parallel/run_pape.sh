#!/bin/sh
# test pape
#export PYTHONPATH=/mnt/lustre/zhuyuanhao/pape/refactor/PAPExtension/:$PYTHONPATH
p=$1
g=$(($2<8?$2:8))
export NCCL_IB_HCA=mlx5_1
OMPI_MCA_mpi_warn_on_fork=0 srun -x SH-IDC1-10-198-4-[147] --mpi=pmi2 -p $p --gres=gpu:$g -n $2 --ntasks-per-node=$g \
python -u benchmark_pape.py --benchmark --max_iter=500 --bucket_size=4 -a alexnet_bn -b 64 /mnt/lustre/share/images
