# Run yolov3 on 1984 and 1424

## 1984

**本地安装环境：**

source /mnt/lustre/share/platform/env/pt1.7.0

拷贝/mnt/luster/share_data/jiangyongjiu/yolov3/Arial.ttf & Arial.Unicode.ttf文件到当前目录

pip install -r requirements

**执行：**

sh train_1984.sh pat_rd 1 --noval

--noval表示不执行test，相关参数可自选添加

## 1424_200 (mlu290)

**环境配置：**
source /mnt/lustre/share/platform/env/pat2.0_dev_gcc5.4
编译parrots，分支临时使用yj/camb_yolov3
torchvision使用 /mnt/lustre/share/jiangyongjiu1/dev/parrots.torchvision
设置PYTHONPATH： export PYTHONPATH=path_to_senseparrots:/mnt/lustre/share/jiangyongjiu1/dev/parrots.torchvision
**执行：**
sh train_mlu.sh camb_mlu290 1 --batch-size=8 --noval