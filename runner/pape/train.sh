mkdir -p log/pape
now=$(date +"%Y%m%d_%H%M%S")

ROOT=.
pyroot=$ROOT/models/parrots.example
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/pape/${name}.yaml

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=${name} \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g --job-name=$4 \
    python -u $ROOT/models/parrots.example/tools/main.py --config ${cfg} \
    2>&1 | tee log/pape/train_$4.log-$now
