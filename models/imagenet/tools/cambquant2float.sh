#!/bin/sh
config=$1
save=$2
echo $save
if [ "$save" = "saveModel" ];
then
    echo "Here run for save float model convert from quantize model!"
    srun -p camb_mlu290 -n8 --gres=mlu:8 python modeltrans.py \
        --config ../configs/${config}.yaml --test --quantify \
        --quant2float --launcher slurm
elif [ "$save" = "saveData" ];
then
    echo "Here run for save input and output data of float model!"
    srun -p camb_mlu290 -n1 --gres=mlu:1 python modeltrans.py \
        --config ../configs/${config}.yaml --test --quantify \
        --quant2float --saveInOut --launcher slurm
else
    echo "Wrong mode for the script!"
fi

######################################README##################################
# $1: is the name of model config with no sufix
# $2: is the mode of this scripts, "saveModel" represent executing the 
# "saveModel": trans from quanitize model to float model. Also run test for quantize and float model once;
# "saveData": exectuing the float model for save input and output data.
