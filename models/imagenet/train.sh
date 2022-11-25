mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
cfg=configs/$2.yaml
port=`expr $RANDOM % 10000 + 10000`
mpirun -np $1 python -u main.py --launcher mpi --port $port --config $cfg \
    2>&1 | tee log/train_$1_mlu_$2.log-$now