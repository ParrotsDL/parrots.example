mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

# export PYTHONPATH=/yourpathto/PAPExtension:$PYTHONPATH

cfg=$1
jobname=$2
gpus=$3

g=$(($gpus<8?$gpus:8))

OMPI_MCA_mpi_warn_on_fork=0 GLOG_vmodule=MemcachedClient=-1 \
mpirun -np $gpus \
    python -u main.py --config $cfg \
    2>&1 | tee log/train_$jobname.log-$now
