mkdir -p log/mmpose
now=$(date +"%Y%m%d_%H%M%S")

ROOT=.
pyroot=$ROOT/models/mmpose
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$3
g=$(($2<8?$2:8))
cfg=$ROOT/configs/mmpose/${name}.py
work_dir=$pyroot/work_dirs/${name}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 --job-name=mmpose_${name}\
    --gres=gpu:$g -n$2 --ntasks-per-node=$g \
    --kill-on-bad-exit=1 \
    python -u $ROOT/models/mmpose/tools/train.py \
        $cfg \
        --work-dir=${work_dir}
        --launcher="slurm" \
        ${EXTRA_ARGS} \
    2>&1|tee log/star/train_${name}.log-$now
