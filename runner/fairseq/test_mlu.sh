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
log_path=log/fairseq/

# test model
# using: sh runner/fairseq/test_mlu.sh camb_mlu290 ckpt_path
srun -p $1 -n1 --gres=mlu:1 --ntasks-per-node=1 --job-name=fairseq_test \
${SRUN_ARGS} python models/fairseq/fairseq_cli/generate.py $BINARY_DATA_PATH \
--path $2 --beam 5 --batch-size 128 --remove-bpe --eval-bleu --device gpu \
--sacrebleu --quiet ${EXTRA_ARGS} 2>&1 | tee ${log_path}/fairseq_mlu290_test_${T}.log
