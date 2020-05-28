#!/bin/sh

p=$1
g=$(($2<8?$2:8))
mode=$3
OMPI_MCA_mpi_warn_on_fork=0 srun --mpi=pmi2 -p $p --gres=gpu:$g -n $2 --ntasks-per-node=$g \
python -u ./benchmark.py \
--benchmark --parallel=$mode --max_iter=200 --bucket_size=4 -a resnet50 -b 64 /mnt/lustre/share/images
