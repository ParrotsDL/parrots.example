# test parrots save and parrots load
python save_parrots_pth.py

python parrots_load_and_check.py

# load pytorch data and test parrots load
unzip  ./pytorch_data.zip

python pytorch_data_and_parrots_load.py

rm -rf *put* && rm -rf net* && rm -rf __pycache__
rm -rf pytorch_data/
