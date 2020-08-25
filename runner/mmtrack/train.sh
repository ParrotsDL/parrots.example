mkdir -p log/mmtrack
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/mmtrack
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/mmtrack/${name}.py

mkdir -p $pyroot/work_dirs/${name}_${now}
work_dir=$pyroot/work_dirs/${name}_${now}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=example_${name} \
    --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
    python -u $ROOT/models/mmtrack/tools/train.py ${cfg} \
    --work-dir=${work_dir} ${EXTRA_ARGS} --launcher=slurm \
    2>&1 | tee log/mmtrack/train_${name}.log-$now

