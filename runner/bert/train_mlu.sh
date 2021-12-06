mkdir -p log/bert
now=$(date +"%Y%m%d_%H%M%S")
set -x

export CNCL_MLULINK_TIMEOUT_SECS=-1
export SQUAD_DIR=/mnt/lustre/share/datasets/nlp/SQuAD

if [ -z $SQUAD_DIR ]; then
    echo "[ERROR] Please set SQUAD_DIR"
    exit 1
fi

run_cmd="python ./models/bert/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --do_train \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --per_gpu_train_batch_size 16 \
    --learning_rate 4e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./models/bert/bert_base_cased_ddp_from_scratch_$now \
    --max_steps -1 \
    --do_eval   \
    "

g=$(($2<8?$2:8))
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
SRUN_ARGS=${SRUN_ARGS:-""}

if [[ $3 =~ "sync" ]]; then
    PARROTS_EXEC_MODE=SYNC OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=bert_train \
        --gres=mlu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        $run_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee log/bert/train_bert.log-$now
else
    OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
    srun --mpi=pmi2 -p $1 --job-name=bert_train \
        --gres=mlu:$g -n$2 --ntasks-per-node=$g  ${SRUN_ARGS} \
        $run_cmd \
        ${EXTRA_ARGS} \
        2>&1 | tee log/bert/train_bert.log-$now
fi