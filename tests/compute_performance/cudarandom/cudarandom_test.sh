size=${2-'500000'}
rsize=${3-'500000'}
srun -p $1 --gres=gpu:8 \
python -u cudarandom_test.py --size=$size --randperm-size=$rsize
