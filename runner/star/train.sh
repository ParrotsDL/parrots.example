mkdir -p log/star
now=$(date +"%Y%m%d_%H%M%S")

ROOT=.
pyroot=$ROOT/models/STAR
papePath=/mnt/lustre/hanyachao/star_test/pape_pt1.4::$PYTHONPATH
export PYTHONPATH=$pyroot:$papePath

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/star/${name}.py

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=${name}\
    --gres=gpu:$g -n$2 --ntasks-per-node=$g \
    --kill-on-bad-exit=1 \
    python -u $ROOT/models/STAR/tools/train.py \
        $cfg \
        --gpus $g \
        --exp_id=$name \
        --use_pape=1 \
        --pavi \
    2>&1|tee log/star/train_${name}.log-$now
