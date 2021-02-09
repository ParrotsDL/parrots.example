# test parrots and pytorch 
parrots=$1
pytorch=$2


source deactivate
source $parrots
srun -p $3 --gres=gpu:1 python cudastack_test.py

source deactivate
source $pytorch
srun -p $3 --gres=gpu:1 python cudastack_test.py

python cudastack_compare.py

rm -f *.pkl
source deactivate
