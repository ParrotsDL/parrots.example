#!/bin/bash



mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
name=$1
cfg=./configs/${name}.yaml
python -u ./main.py --config ${cfg}
