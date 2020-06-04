work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n32 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u main.py --config $work_path/config.yaml \
 --bn_sync_mode=sync \
 --bn_group_size=1 \
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover
 #--evaluate
 #--bn_sync_mode=sync \
 #--bn_group_size=8
 #--fake \
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover
