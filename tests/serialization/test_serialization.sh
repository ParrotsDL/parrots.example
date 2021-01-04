# test parrots save and pytorch load
parrots=$1
pytorch=$2

source deactivate
source $parrots
python save_parrots_pth.py

source deactivate
source $pytorch
python pytorch_load_and_check.py

# test pytorch save and parrots load
rm -rf *put* && rm -rf net* && rm -rf __pycache__
python save_parrots_pth.py

source deactivate
source $parrots
python parrots_load_and_check.py

rm -rf *put* && rm -rf net* && rm -rf __pycache__
