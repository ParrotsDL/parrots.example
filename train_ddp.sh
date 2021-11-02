#!/bin/bash
CUR_DIR=$(cd $(dirname $0);pwd)
if [ -z $MLU_VISIBLE_DEVICES ];then
    export MLU_VISIBLE_DEVICES=0,1,2,3
fi

train_iters=-1

if [ $1 -eq 0 ]; then
    train_iters=2
    num_train_epochs=1
elif [ $1 -eq 3 ]; then
    export MLU_ADAPTIVE_STRATEGY_COUNT=100
    num_train_epochs=1
else
    num_train_epochs=2
fi

nproc_per_node=4
nnodes=1
runcmd="python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --nnodes=${nnodes} run_squad.py  --model_type bert   --model_name_or_path bert-base-cased   --do_train  --train_file $SQUAD_DIR/train-v1.1.json   --predict_file $SQUAD_DIR/dev-v1.1.json   --per_gpu_train_batch_size 16   --learning_rate 4e-5   --num_train_epochs ${num_train_epochs}   --max_seq_length 384   --doc_stride 128 --output_dir bert_base_cased_ddp_from_scratch --max_steps ${train_iters}"

if [ $1 -ne 0 ]; then
    runcmd="$runcmd --do_eval"
fi

pushd $CUR_DIR
eval "${runcmd}"
if [ $1 -eq 3 ]; then
  sed -i "s/dev_num:-/dev_num:`expr $nproc_per_node \* $nnodes`/" ${BENCHMARK_LOG}
  sed -i "s/dist_mode:-/dist_mode:DDP/" ${BENCHMARK_LOG}
fi
popd
