#!/usr/bin/env bash
export PYTHONPATH=$ROOT:models/Pattern/:$PYTHONPATH
SRUN_ARGS=${SRUN_ARGS:-""}
now=$(date +"%Y%m%d_%H%M%S")
GLOG_logtostderr=-1 GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
    srun --mpi=pmi2 --job-name $1 --partition=$6 -n$2 --gres=gpu:$3 --ntasks-per-node=$4 ${SRUN_ARGS} \
        python -u models/Pattern/tools/dist_train.py --config $5 --now $now $7

# resume training
#GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
#    srun --mpi=pmi2 --job-name $1 --partition=$1 -n1 --gres=gpu:1 --ntasks-per-node=1 \
#        python -u tools/dist_train.py --config $2 --now $now --resume-opt
