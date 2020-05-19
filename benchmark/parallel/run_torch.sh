#!/bin/sh

p=$1
g=$(($2<8?$2:8))
OMPI_MCA_mpi_warn_on_fork=0 srun -x SH-IDC1-10-198-4-[147] --mpi=pmi2 -p $p --gres=gpu:$g -n $2 --ntasks-per-node=$g \
python -u benchmark_torch.py --benchmark --max_iter=500 --disable_broadcast_buffers -a resnet50 -b 64 /mnt/lustre/share/images
