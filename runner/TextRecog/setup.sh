#!/bin/bash

source /usr/local/env/pat_latest

export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0

cd models/TextRecog
cd pytorch-ctc/pytorch-ctc-0.3.2
mkdir build
cd build
cmake ..
make
export WARP_CTC_PATH=`pwd`
cd ../parrots_binding
pip install -v -e . --user
cd ../../../../../