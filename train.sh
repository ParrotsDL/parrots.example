#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
PYTHON_VERSION=`python -c 'import sys; version=sys.version_info[:3]; \
                print("{0}.{1}.{2}".format(*version))'`
python_version=${PYTHON_VERSION:0:1}

export MLU_VISIBLE_DEVICES=0

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [0|1|2|3] [-1|0|1|2|3]"
    echo "|  parameter1: 0)precheckin, 1)daily, 2)weekly, 3)benchmark"
    echo "|  parameter2: -1)no cnmix, 0)O0, 1)O1, 2)O2, 3)O3."
    echo "|  eg. ./train.sh 0 -1"
    echo "|      which means running bert weekly without cnmix on single card."
    echo "|  eg. ./train.sh 0 1"
    echo "|      which means running bert weekly with cnmix O1 mode on single card."
    echo "-------------------------------------------------------------"
}

if [[ $1 =~ ^[0-3]{1}$ && $2 =~ ^-?[0-3]{1}$ && $2 -ge -1 ]]; then
    echo "Parameters Exact."
else
    echo "[ERROR] Unknown Parameter."
    usage
    exit 1
fi

mode=$1
train_iters=-1

if [ $mode -eq 0 ]; then
    train_iters=2
    num_train_epochs=1
elif [ $mode -eq 3 ]; then
    export MLU_ADAPTIVE_STRATEGY_COUNT=100
    num_train_epochs=1
else
    num_train_epochs=2
fi

# Use cnmix
cnmix_level=""
if [[ $2 -ne -1 ]]; then
    cnmix_level="O${2}"
fi

model_type="bert"
lr="3e-5"
batch_size="16"
max_seq_length="384"
doc_stride="128"
output_dir="bert_base_cased_from_scratch/"
train_file="${SQUAD_DIR}/train-v1.1.json"
predict_file="${SQUAD_DIR}/dev-v1.1.json"

run_train() {
    echo "-----------------------------------------------------------------"
    echo -e "python_version : $PYTHON_VERSION,\nmodel_name : $model_type,\nbatch_size : $batch_size,\nlr : $lr,\neopchs : $epochs."
    run_cmd="python	${CUR_DIR}/run_squad.py \
        --model_type $model_type	\
        --model_name_or_path bert-base-cased	\
        --do_train	\
        --train_file ${train_file}	\
        --predict_file ${predict_file}	\
        --per_gpu_train_batch_size ${batch_size}	\
        --learning_rate ${lr}	\
        --num_train_epochs ${num_train_epochs}	\
        --max_seq_length ${max_seq_length}	\
        --doc_stride ${doc_stride}	\
        --max_steps ${train_iters}	\
        --output_dir ${output_dir}	\
        --overwrite_output_dir"
    if [[ ${cnmix_level} =~ ^O[0-3]{1}$ ]]; then
        echo "cnmix_level $cnmix_level"
        run_cmd="$run_cmd --cnmix --fp16_opt_level ${cnmix_level}"
    fi
    if [[ $mode -eq 1 || $mode -eq 2 ]];then
        run_cmd="$run_cmd --do_eval"
    fi
    echo "$run_cmd"
    eval "$run_cmd" 
}

pushd $CUR_DIR
run_train
popd

