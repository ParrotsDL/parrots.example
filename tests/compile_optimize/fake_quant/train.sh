mkdir -p log/
now=$(date +"%Y%m%d_%H%M%S")

ROOT=../../..
modelroot=$ROOT/models/parrots.example/models/imagenet
quantroot=$ROOT/tests/compile_optimize/fake_quant/Famicom
export PYTHONPATH=$modelroot:$quantroot:$PYTHONPATH

cfg=$1
jobname=$2
partition=$3
gpus=$4

g=$(($gpus<8?$gpus:8))

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $partition --job-name=$jobname \
    --gres=gpu:$g -n$gpus --ntasks-per-node=$g \
    python -u $ROOT/tests/compile_optimize/fake_quant/fake_quant.py --config $cfg \
    2>&1 | tee log/train_$jobname.log-$now
