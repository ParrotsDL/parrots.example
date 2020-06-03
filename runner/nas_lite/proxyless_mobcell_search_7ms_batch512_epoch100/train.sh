mkdir -p log/springnas-lite/

T=`date +%m%d%H%M`
ROOT=.

# add springnas_lite lib without install
pyroot=$ROOT/models/springnas-lite
export PYTHONPATH=$pyroot:$PYTHONPATH

g=$(($2<8?$2:8))
name=$3
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g --cpus-per-task=5 \
python models/springnas-lite/imagenet-example-automl/main.py \
 --config $ROOT/configs/springnas-lite/proxyless_mobcell_search_7ms_batch512_epoch100/config.yaml \
 --bn_sync_mode=sync \
 --bn_group_size=$2 \
 --sync ${EXTRA_ARGS} \
 2>&1 | tee log/springnas-lite/${name}.log-$T
 # TODO(shiguang): use async mode
 #--evaluate
 #--bn_sync_mode=sync \
 #--bn_group_size=8
 #--fake \
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover
