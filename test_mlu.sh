now=$(date +"%Y%m%d_%H%M%S")
set -x

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z $DATASET_PATH ]; then
    echo "[ERROR] Please set DATASET_PATH"
    exit 1
fi

if [ -z $MODEL_PATH ]; then
    echo "[ERROR] Please set MODEL_PATH"
    exit 1
fi

run_cmd="python -u eval.py \
    --log_path $CUR_DIR/logs/eval.txt \
    --pretrained $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --vocab_path $VOCAB_PATH \
    "

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

if [[ $3 =~ "sync" ]]; then
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=transformer_train \
        --gres=mlu:1 -n1 --ntasks-per-node=1  ${SRUN_ARGS} \
        $run_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee logs/test_transformer.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=transformer_train \
        --gres=mlu:1 -n1 --ntasks-per-node=1  ${SRUN_ARGS} \
        $run_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee logs/test_transformer.log-$now
fi
