#/bin/sh
if [ ! -d "log/fairseq/" ]; then
  mkdir -p log/fairseq/
fi
T=`date +%m%d%H%M%S`
CUR_DIR=$(cd $(dirname "$pwd");pwd)

export PYTHONPATH=models/fairseq/:$PYTHONPATH
export BINARY_DATA_PATH=/mnt/lustre/share/datasets/nlp/wmt16_en_de_bpe32k

# extra param of training
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
SRUN_ARGS=${SRUN_ARGS:-""}
saved_path="fairseq_ckpt"
log_path=log/fairseq/

# train model
srun -p $1 -n$2 --gres=mlu:$2 --ntasks-per-node=$2 \
--job-name=fairseq ${SRUN_ARGS} python models/fairseq/fairseq_cli/train.py \
$BINARY_DATA_PATH --arch transformer_wmt_en_de_big \
--share-all-embeddings --dropout 0.3 --weight-decay 0.0 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9, 0,98)' --clip-norm 0.0 --lr 0.0001 \
--min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 4000 --max-tokens 4096 --log-interval 10 \
--keep-interval-updates 20 --steps_per_epoch 4751 \
--save-dir $saved_path --device gpu --distributed-backend cncl \
${EXTRA_ARGS} 2>&1 | tee ${log_path}/fairseq_mlu290_train_${T}.log

# test model
srun -p $1 -n1 --gres=mlu:1 --ntasks-per-node=1 --job-name=fairseq_test \
${SRUN_ARGS} python models/fairseq/fairseq_cli/generate.py $BINARY_DATA_PATH \
--path ${saved_path}/checkpoint_last.pt --beam 5 --batch-size 128 \
--remove-bpe --eval-bleu --device gpu --sacrebleu --quiet ${EXTRA_ARGS} \
2>&1 | tee ${log_path}/fairseq_mlu290_test_${T}.log
