#!/bin/bash
set -x
set -o pipefail
set -e

# 0. check the most important SMART_ROOT
echo  "!!!!!SMART_ROOT is" $SMART_ROOT
if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 1. set env_path
if [[ $PWD =~ "example" ]]
then 
    pyroot=$PWD
else
    pyroot=$PWD/example
fi
echo $pyroot

# 2. build file folder for save log and set time
mkdir -p algolib_gen/example
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env variables
export FRAME_NAME=example # customize for each frame
export MODEL_NAME=$3
export PYTHONPATH=$pyroot:$PYTHONPATH
export PARROTS_DEFAULT_LOGGER=FALSE
# add path for mole
export PYTHONPATH=$pyroot/models/imagenet/:$PYTHONPATH


# 4. init_path
export PYTHONPATH=${SMART_ROOT}:$PYTHONPATH
export PYTHONPATH=$SMART_ROOT/common/sites/:$PYTHONPATH # necessary for init

# 5. build necessary parameter
cfg=$pyroot/algolib/configs/${MODEL_NAME}.yaml
partition=$1
g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

# 6. set port and choice model
port=`expr $RANDOM % 10000 + 20000`
mkdir -p algolib_gen/example/${MODEL_NAME}/

# 7. run model
if [[ $3 =~ "sync" ]]; then
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $partition --job-name=example_${MODEL_MODEL_NAME} \
        --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        python -u $pyroot/models/imagenet/main.py --config ${cfg} \
        --save_path=algolib_gen/example/${MODEL_NAME} --port=$port \
        ${EXTRA_ARGS} \
        2>&1 | tee algolib_gen/example/${MODEL_NAME}/train_${MODEL_NAME}.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $partition --job-name=example_${MODEL_NAME} \
        --gres=gpu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS}  \
        python -u $pyroot/models/imagenet/main.py --config ${cfg} \
        --save_path=algolib_gen/example/${MODEL_NAME} --port=$port \
        ${EXTRA_ARGS} \
        2>&1 | tee algolib_gen/example/${MODEL_NAME}/train_${MODEL_NAME}.log-$now
fi
