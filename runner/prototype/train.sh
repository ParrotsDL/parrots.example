mkdir -p log/prototype
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/prototype
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/prototype/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=prototype_${name} \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
    python -u -m prototype.solver.cls_solver --config ${cfg} \
    ${EXTRA_ARGS} \
    2>&1 | tee log/prototype/train_${name}.log-$now
