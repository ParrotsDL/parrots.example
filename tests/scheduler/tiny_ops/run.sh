srun -p $1 --gres=gpu:8 --exclusive python -u bbox.py
