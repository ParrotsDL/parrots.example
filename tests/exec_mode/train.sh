mkdir -p log/example
now=$(date +"%Y%m%d_%H%M%S")

ROOT=.
pyroot=$ROOT/models/parrots.example
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/example/${name}.yaml
mode=$4

if [ ! $mode ]; then
    echo "mode is not specified!"
    exit -1
fi

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
PARROTS_EXEC_MODE=${mode} \
srun --mpi=pmi2 -p $1 --job-name=${name} \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g \
    python -u $ROOT/models/parrots.example/models/imagenet/main.py --config ${cfg} \
    2>&1 | tee log/example/train_${name}_${mode}.log-$now
