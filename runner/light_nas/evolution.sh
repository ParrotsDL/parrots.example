mkdir -p log/light_nas
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/light_nas
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/light_nas/${name}.yaml

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
step='evolution'

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=light_nas_${name} \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS}\
    python -m main --config ${cfg} --step ${step} \
    ${EXTRA_ARGS} \
    2>&1 | tee log/light_nas/train_${step}_${name}.log-$now
