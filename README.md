# Run yolov3 on 1984 and 1424

## 1984 (v100)

**本地安装环境：**

source /mnt/lustre/share/platform/env/pt1.7.0

拷贝/mnt/luster/share_data/jiangyongjiu/yolov3/Arial.ttf & Arial.Unicode.ttf文件到当前目录

pip install -r requirements

**执行：**

sh train_1984.sh pat_rd 1

相关参数可根据train.py脚本自选添加到train_1984.sh中

## 1424_200 (mlu290)

**环境配置：**
source /mnt/lustre/share/platform/env/pat2.0_dev_gcc5.4
编译parrots，分支使用openinfra/dev/v1.0
设置PYTHONPATH： export PYTHONPATH=path_to_senseparrots:$PYTHONPATH
若echo $LC_ALL为空，需要设置export LC_ALL=en_US.UTF-8

**执行：**
sh train_mlu.sh camb_mlu290 1
相关参数可根据train.py参数列表自选添加到train_mlu290.sh中
