#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 run_squad.py \
python run_squad.py \
 --model_type bert --model_name_or_path bert-base-cased \
--do_train --train_file  $SQUAD_DIR/train-v2.0.json  \
--predict_file $SQUAD_DIR/dev-v2.0.json  \
--per_gpu_train_batch_size 16 \
--learning_rate 4e-5 --num_train_epochs 2\
 --max_seq_length 384 --doc_stride 128 --output_dir bert_base_cased_ddp_from_scratch --max_steps -1 --do_eval
#\ --cnmix --fp16_opt_level o0
