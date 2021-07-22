set -x
# 1. build file folder for save log
mkdir -p algolib_log/example
now=$(date +"%Y%m%d_%H%M%S")

# 2. set env 
ROOT=.
pyroot=$ROOT/algolib/example
export PYTHONPATH=$pyroot:$PYTHONPATH
export MODEL_NAME=$3

# 3. build necessary parameter
partition=$1
g=$(($2<8?$2:8))
MODEL_NAME=$3
cfg=$ROOT/algolib/example/algolib/configs/${MODEL_NAME}.yaml

# 4. build optional parameter
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}


if [[ $3 =~ "sync" ]]; then
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $partition --job-name=example_${MODEL_MODEL_NAME} \
        --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        python -u $pyroot/models/imagenet/main.py --config ${cfg} \
        ${EXTRA_ARGS} \
        2>&1 | tee algolib_log/example/train_${MODEL_NAME}.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=example_${MODEL_NAME} \
        --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS}  \
        python -u $pyroot/models/imagenet/main.py --config ${cfg} \
        ${EXTRA_ARGS} \
        2>&1 | tee algolib_log/example/train_${MODEL_NAME}.log-$now
fi