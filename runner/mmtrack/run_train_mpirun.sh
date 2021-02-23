mkdir -p log/mmtrack
now=$(date +"%Y%m%d_%H%M%S")
set -x
ROOT=.
pyroot=$ROOT/models/mmtrack
export PYTHONPATH=$pyroot:$PYTHONPATH

name=$1

cfg=$ROOT/configs/mmtrack/${name}.py

mkdir -p $pyroot/work_dirs/${name}
work_dir=$pyroot/work_dirs/${name}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

python -u $ROOT/models/mmtrack/tools/train.py ${cfg} \
    --work-dir=${work_dir} ${EXTRA_ARGS} --launcher=mpi \
    2>&1 | tee log/mmtrack/train_${name}.log-$now

