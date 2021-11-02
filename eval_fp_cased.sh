export SQUAD_DIR=./squad_v1.1
python run_squad_fp.py   --model_type bert   --model_name_or_path cased --do_eval   --train_file $SQUAD_DIR/train-v1.1.json   --predict_file $SQUAD_DIR/dev-v1.1.json   --per_gpu_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2.0   --max_seq_length 384   --doc_stride 128  --output_dir cased

