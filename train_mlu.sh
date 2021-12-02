mkdir -p logs
now=$(date +"%Y%m%d_%H%M%S")
set -x

CUR_DIR=$(cd $(dirname $0);pwd)

if [ -z $DATASET_PATH ]; then
    echo "[ERROR] Please set DATASET_PATH"
    exit 1
fi

if [ -z $VOCAB_PATH ]; then
    echo "[ERROR] Please set VOCAB_PATH"
    exit 1
fi

num_epochs=20
iterations=-1
print_freq=50
dropout_rate=0.0
batch_size=$(($2>=4?8:32))

run_cmd="python -u train.py \
    --log-path $CUR_DIR/logs \
    --num_epochs $num_epochs \
    --iterations $iterations \
    --print-freq ${print_freq} \
    --dropout_rate $dropout_rate \
    --dataset-path $DATASET_PATH \
    --vocab_path $VOCAB_PATH \
    --batch-size $batch_size  \
    "

g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

if [[ $3 =~ "sync" ]]; then
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=transformer_train \
        --gres=mlu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        $run_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee logs/train_transformer.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=transformer_train \
        --gres=mlu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        $run_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee logs/train_transformer.log-$now
fi
