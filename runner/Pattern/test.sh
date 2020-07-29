#!/usr/bin/env bash
export PYTHONPATH=$ROOT:models/Pattern/:$PYTHONPATH
# srun once
#GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 \
#    srun --mpi=pmi2 --job-name test --partition=Metric -n1 --gres=gpu:1 --ntasks-per-node=1 \
#        python -u tools/dist_test.py --config $1 -e --start=$2 --strip=$3 --end=$4

# srun repeatedly
start_iter=$3
step=5000
end_iter=$4
model_folder=$2
while test $start_iter -le $end_iter; do
    model_name="iter_${start_iter}_ckpt.pth.tar"
    echo "testing: "${model_folder}$model_name
    if [ -f ${model_folder}${model_name} ]; then
        GLOG_vmodule=MemcachedClient=-1 OMPI_MCA_btl_smcuda_use_cuda_ipc=0 OMPI_MCA_mpi_warn_on_fork=0 \
            srun --mpi=pmi2 --job-name test --partition=$5 -n1 --gres=gpu:1 --ntasks-per-node=1 \
                python -u models/Pattern/tools/dist_test.py --config $1 -e --model-name=$model_name
        ((start_iter+=step))
    else
        echo "sleep"
        sleep 30m
    fi
done
