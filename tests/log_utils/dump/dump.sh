#! /bin/bash

python3 setup.py clean
rm ~/.local/lib/python3.6/site-packages/dump_extension* -rf
srun -p $1 --gres=gpu:1 python3 setup.py install --user
PARROTS_COREDUMP=ON srun -p $1 --gres=gpu:1 python3 seg.py
