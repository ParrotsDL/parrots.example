set -x
set -o pipefail
set -e

if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

if [ -x "$SMART_ROOT/submodules" ];then
    submodules_root=$SMART_ROOT
else    
    submodules_root=$PWD
fi

# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/example

# 2. set time
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env
path=$PWD
if [[ "$path" =~ "submodules" ]]
then 
    pyroot=$submodules_root/example
else
    pyroot=$submodules_root/submodules/example
fi
export FRAME_NAME=example # customize for each frame
export MODEL_NAME=$3
cfg=$pyroot/algolib/configs/${MODEL_NAME}.yaml
export PYTHONPATH=$pyroot:$PYTHONPATH

# init_path
export PYTHONPATH=$SMART_ROOT/common/sites/:$PYTHONPATH # necessary for init

# 4. build necessary parameter
partition=$1
g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

port=`expr $RANDOM % 10000 + 20000`

# 5. model choice
mkdir -p algolib_gen/example/${MODEL_NAME}/
export PARROTS_DEFAULT_LOGGER=FALSE
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
