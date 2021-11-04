mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

model=$1
partition=$2
gpus=$3

cfg=configs/${model}.yaml
jobname=${model}_example
g=$(($gpus<8?$gpus:8))

srun -p $partition --job-name=$jobname \
    --gres=mlu:$g -n$gpus python -u main.py --config $cfg \
    2>&1 | tee log/train_${jobname}_${gpus}cards.log-$now