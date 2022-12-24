mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
set -x

CUR_DIR=$(cd $(dirname $0);pwd)

export DATASET_PATH=/mnt/lustre/share_data/PAT/datasets/nlp/corpora/
export VOCAB_PATH=./models/transformer/data/IWSLT/preprocessed/

if [ -z $DATASET_PATH ]; then
    echo "[ERROR] Please set DATASET_PATH"
    echo "  IWSLT16: /mnt/lustre/share/datasets/nlp/corpora/"
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
batch_size=$(($1>=4?8:32))

run_cmd="python -u ./models/transformer/train.py \
    --log-path ./log/transformer \
    --num_epochs $num_epochs \
    --iterations $iterations \
    --print-freq ${print_freq} \
    --dropout_rate $dropout_rate \
    --dataset-path $DATASET_PATH \
    --vocab_path $VOCAB_PATH \
    --ckp-path ./models/transformer/checkpoint \
    --batch-size $batch_size \
    "

g=$(($1<8?$1:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1
mpirun -np $1 $run_cmd --launcher mpi ${EXTRA_ARGS}  2>&1 | tee log/transformer/train_transformer.log-$now
