#!/bin/sh

# test pape
#export PYTHONPATH=/mnt/lustre/zhuyuanhao/pape/refactor/PAPExtension/:$PYTHONPATH
p=$1
g=$(($2<8?$2:8))
#OMPI_MCA_mpi_warn_on_fork=0 srun --mpi=pmi2 -p $p --gres=gpu:$g -n $2 --ntasks-per-node=$g \
#python -u benchmark_pape.py --benchmark --max_iter=500 --bucket_size=4 -a resnet50 -b 64 /mnt/lustre/share/images
OMPI_MCA_mpi_warn_on_fork=0 srun --mpi=pmi2 -p $p --gres=gpu:$g -n $2 --ntasks-per-node=$g \
python -u benchmark_pape.py --bucket_size=4 -a resnet50 -b 64 /mnt/lustre/share/images
