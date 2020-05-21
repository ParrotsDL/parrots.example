mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=/mnt/lustre/zhuyuanhao/example/PAPExtension:$PYTHONPATH

cfg=$1
jobname=$2
partition=$3
gpus=$4

g=$(($gpus<8?$gpus:8))

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $partition --job-name=$jobname \
    --gres=gpu:$g -n$gpus --ntasks-per-node=$g \
    python -u main_iter.py --config $cfg --test --checkpoint /mnt/lustre/zhuyuanhao/example/parrots.example/models/imagenet/checkpoints/resnet50d-bs128/resnet50d_iter_250000.pth \
    2>&1 | tee log/test_resnet50d_bs128_iter_250000.log-$now
