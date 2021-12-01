mkdir -p log/example
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/parrots.example
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/example/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

if [[ $4 =~ "sync" ]]; then
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=${name} \ 
        --gres=dcu:$g -n$2 --ntasks-per-node=$g  --cpus-per-task 8 ${SRUN_ARGS} \
        numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 \
        python -u $ROOT/models/parrots.example/models/imagenet/main.py --config ${cfg} \
        ${EXTRA_ARGS} \
        2>&1 | tee log/example/train_${name}.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=${name} \ 
        --gres=dcu:$g -n$2 --ntasks-per-node=$g --cpus-per-task 8 ${SRUN_ARGS} \
        numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 \
        python -u $ROOT/models/parrots.example/models/imagenet/main.py --config ${cfg} \
        ${EXTRA_ARGS} \
        2>&1 | tee log/example/train_${name}.log-$now
fi
