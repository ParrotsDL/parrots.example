mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

function helper
{
    echo "Command:"
    echo "  sh train_mlu.sh [0] [1] [2]"
    echo "  args:"
    echo "      [0] model name"
    echo "      [1] partition name"
    echo "      [2] card number"
    echo "  eg. sh train_mlu.sh googlenet camb_mlu290 8"
    echo "      which means training googlenet model on camb_mlu290 partition with 8 cards."
}

model=$1
partition=$2
gpus=$3

if [ $# -ne 3 ]; then
    echo "[ERROR] need 3 arguments."
    helper
    exit 1
fi

cfg=configs/${model}.yaml
jobname=${model}_example
g=$(($gpus<8?$gpus:8))

srun -p $partition --job-name=$jobname \
    --gres=mlu:$g -n$gpus python -u main.py --config $cfg \
    2>&1 | tee log/train_${jobname}.log-$now

if [ ${PIPESTATUS[0]} -eq 1 ]; then
    echo "[ERROR] training failed."
    helper
    exit 1
fi