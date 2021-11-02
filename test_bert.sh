#!/bin/bash
CUR_DIR=$(cd $(dirname $0);pwd)
PYTHON_VERSION=`python -c 'import sys; version=sys.version_info[:3]; \
                print("{0}.{1}.{2}".format(*version))'`
python_version=${PYTHON_VERSION:0:1}

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [0|1|2] [0|1|2|3] [-1|0|1|2|3] [0|1]"
    echo "|  parameter1: 0)single card, 1)multiprocessing-distributed, 2)horovod."
    echo "|  parameter2: 0)precheckin, 1)daily, 2)weekly, 3)benchmark"
    echo "|  parameter3: -1)no cnmix, 0)O0, 1)O1, 2)O2, 3)O3."
    echo "|  parameter4: 0)MLU, 1)GPU"
    echo "|  eg. ./train.sh 0 0 -1 0"
    echo "|      which means running bert precheckin without cnmix on single MLU card."
    echo "-------------------------------------------------------------"
}

if [[ $1 =~ ^[0-2]{1}$ && $2 =~ ^[0-3]{1}$ && $3 =~ ^-?[0-3]{1}$ && $3 -ge -1 && $4 =~ ^[0-1]{1}$ ]]; then
    echo "Parameters Exact."
else
    echo "[ERROR] Unknown Parameter."
    usage
    exit 1
fi

distributed=$1
mode=$2
# Use cnmix
cnmix_level=""
if [[ $3 -ne -1 ]]; then
    cnmix_level="O${3}"
fi

device_param="mlu"
if [[ $4 -eq 1 ]]; then
    device_param="gpu"
fi

train_iters=-1
eval_iters=-1

if [ $mode -eq 0 ]; then
    train_iters=2
    eval_iters=2
    num_train_epochs=1
elif [ $mode -eq 3 ]; then
    pip install transformers==3.5.0
    bert_checkpoint=~/.cache/torch
    if [ ! -d "${bert_checkpoint}/transformers" ];then
      if [ ! -d ${bert_checkpoint} ];then
        mkdir $bert_checkpoint
      fi
      src="$IMAGENET_TRAIN_CHECKPOINT/bert-base-cased/torch/transformers"
      ln -s $src "${bert_checkpoint}/transformers"
    fi
    export MLU_ADAPTIVE_STRATEGY_COUNT=100
    num_train_epochs=1
else
    num_train_epochs=2
fi


model_type="bert"
lr="3e-5"
batch_size="16"
max_seq_length="384"
doc_stride="128"
output_dir="bert_base_cased_from_scratch/"
train_file="${SQUAD_DIR}/train-v1.1.json"
predict_file="${SQUAD_DIR}/dev-v1.1.json"

use_launch=""
nproc_per_node=4
nnodes=1
if [[ $distributed -eq 1 ]]; then
    use_launch="-m torch.distributed.launch --nproc_per_node=${nproc_per_node} --nnodes=${nnodes}"
    lr="4e-5"
    output_dir="bert_base_cased_ddp_from_scratch/"
fi

run_train() {
    echo "-----------------------------------------------------------------"
    echo -e "python_version : $PYTHON_VERSION,\nmodel_name : $model_type,\nbatch_size : $batch_size,\nlr : $lr,\neopchs : $epochs."
    run_cmd="python $use_launch ${CUR_DIR}/run_squad.py \
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
        --eval_iters ${eval_iters}  \
        --output_dir ${output_dir}	\
        --overwrite_output_dir  \
        --device_param ${device_param}"
    if [[ ${cnmix_level} =~ ^O[0-3]{1}$ ]]; then
        echo "cnmix_level $cnmix_level"
        run_cmd="$run_cmd --cnmix --fp16_opt_level ${cnmix_level}"
    fi
    if [[ $mode -eq 1 || $mode -eq 2 ]]; then
        run_cmd="$run_cmd --do_eval"
    fi
    echo "$run_cmd"
    eval "$run_cmd" 
}

pushd $CUR_DIR
run_train
popd

