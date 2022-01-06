# Run yolov3 on 1984 and 1424

## 1984 (v100)

**本地安装环境：**
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
source /mnt/lustre/share/platform/env/pt1.7.0
cp /mnt/luster/share_data/jiangyongjiu/yolov3/Arial.ttf .
cp /mnt/luster/share_data/jiangyongjiu/yolov3/Arial.Unicode.ttf .
pip install -r requirements
```

**执行：**
```shell
# train
sh tools/train_gpu.sh pat_rd 4

# train with pretrained model
sh tools/train_gpu.sh pat_rd 4 --pretrained_model xxx.pth

# test a pretrained model
sh tools/train_gpu.sh pat_rd 4 --pretrained_model xxx.pth --test
```

相关参数可根据train.py脚本自选增加到启动脚本后增加相应参数，如需要增加保存checkpoint的路径，则执行以下命令
```shell
sh tools/train_gpu.sh pat_rd 4 --saved_path checkpoint
```

## 1424_200 (mlu290)

**环境配置：**
```shell
source /mnt/lustre/share/platform/env/pat2.0_dev_gcc5.4
# compile parrots
git clone git@gitlab.bj.sensetime.com:platform/ParrotsDL/senseparrots.git
cd senseparrots; git checkout openinfra/dev/v1.0;
sh scripts/ci_camb_script.sh mk_camb_mpion; cd python;make -j;
# PYTHONPATH
export PYTHONPATH=path_to_parrots:path_to_torchvision:$PYTHONPATH
```
若echo $LC_ALL为空，需要设置
```shell
export LC_ALL=en_US.UTF-8
```

**执行：**
```shell
# train
sh tools/train_mlu.sh camb_mlu290 4
# train with pretrained model
sh tools/train_mlu.sh camb_mlu290 4 --pretrained_model xxx.pth
# test 
sh tools/test_mlu.sh camb_mlu290 4 --pretrained_model xxx.pth
```
ps: 寒武纪脚本单独对模型做test时，尽量先使用单卡进行测试，多卡使用时容易在创建dataloader时卡住（后期修复）。

若需额外参数，可根据train.py参数列表自选添加到train_mlu290.sh结尾参数中，具体可以参考在v100上训练时增加参数的方法
