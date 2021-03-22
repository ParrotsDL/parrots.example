#!/bin/bash
#!/bin/bash
set -x
source $1

name=$2
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
 
## 下面是未定的存储和pavi使用方案，我先暂时这样写了
if [ "x$OMPI_COMM_WORLD_LOCAL_RANK" == "x0" ]
then
    cp /mnt/lustre/${USER}/petreloss.conf /home/${USER}/petreloss.conf
    cp -r /mnt/lustre/${USER}/.pavi /home/${USER}/.pavi
fi


mkdir -p log/springce_psot/

T=`date +%m%d%H%M%S`
ROOT=.
test_dataset=VOT2018

cfg=$PWD/configs/springce_psot/${name%%_*}/${name#*_}/config.yaml


pyroot=$ROOT/models/springce_psot
if [ ! -d "${pyroot}/testing_dataset/${test_dataset}" ]; then
  ln -s /s3://parrots_model_data/PSOT/ucg.tracking.academic.test/${test_dataset} ${pyroot}/testing_dataset/${test_dataset}
fi

export PYTHONPATH=$pyroot/experiments/${name%%_*}/${name#*_}:$pyroot:$PYTHONPATH

cd $pyroot
python setup.py build_ext --inplace
cd -

python -u -m siamrpn train \
  --cfg=$cfg --test_dataset=${test_dataset} ${EXTRA_ARGS} \
  2>&1 | tee $ROOT/log/springce_psot/train.${name}.log.$T