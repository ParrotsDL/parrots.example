partition=$1
gpu_num=$2
g=$(($2<8?$2:8))
name=$3
shift && shift && shift

ROOT=.
cfg=$ROOT/configs/mild/${name}.yaml

# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$ROOT/models/mild/evaluate/kestrel
export PYTHONPATH=$ROOT/models/mild:$ROOT/models/mild/thirdparty_update:$PYTHONPATH

mkdir -p log/mild
now=$(date +"%Y%m%d_%H%M%S")
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun -K --mpi=pmi2 -p $partition --job-name=mild_${name} \
    --gres=gpu:$g -n$gpu_num --ntasks-per-node=$g  ${SRUN_ARGS}\
    python -u $ROOT/models/mild/run/start.py "cuda" "mimic" $cfg $@ \
    2>&1 | tee log/mild/train_${name}.log-$now
