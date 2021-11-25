mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
set -x

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_PATH=/mnt/lustre/share/datasets/nlp/corpora/
export VOCAB_PATH=./models/transformer/data/IWSLT/preprocessed/
export MODEL_PATH=./models/transformer/checkpoint/model_epoch_10.pth

if [ -z $DATASET_PATH ]; then
    echo "[ERROR] Please set DATASET_PATH"
    echo "  IWSLT16: /mnt/lustre/share/datasets/nlp/corpora/"
    exit 1
fi

if [ -z $VOCAB_PATH ]; then
    echo "[ERROR] Please set VOCAB_PATH"
    echo "  VOCAB_PATH is the path to your preprocessed vocab data."
    echo "  e.g. models/transformer/data/IWSLT/preprocessed/"
    exit 1
fi

if [ -z $MODEL_PATH ]; then
    echo "[ERROR] Please set MODEL_PATH"
    echo "  MODEL_PATH is the path to your checkpoint."
    echo "  e.g. models/transformer/checkpoint/model_epoch_10.pth"
    exit 1
fi

num_epochs=10
iterations=-1
print_freq=50
dropout_rate=0.0

train_cmd="python -u ./models/transformer/train.py \
    --log-path ./log/transformer \
    --num_epochs $num_epochs \
    --iterations $iterations \
    --print-freq ${print_freq} \
    --dropout_rate $dropout_rate \
    --dataset-path $DATASET_PATH \
    --vocab_path $VOCAB_PATH \
    --ckp-path ./models/transformer/checkpoint \
    "

test_cmd="python -u ./models/transformer/eval.py \
    --log_path ./log/transformer/eval.txt \
    --pretrained $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --vocab_path $VOCAB_PATH \
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
        $train_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee log/transformer/train_transformer.log-$now
    
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=transformer_train \
        --gres=mlu:1 -n1 --ntasks-per-node=1  ${SRUN_ARGS} \
        $test_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee log/transformer/test_transformer.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=transformer_train \
        --gres=mlu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        $train_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee log/transformer/train_transformer.log-$now

    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=transformer_train \
        --gres=mlu:1 -n1 --ntasks-per-node=1  ${SRUN_ARGS} \
        $test_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee log/transformer/test_transformer.log-$now
fi
