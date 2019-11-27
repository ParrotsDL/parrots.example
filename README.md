## Parrots2 examples
Image Classification Training in Parrots and Pytorch.

Get detailed documentation in http://confluence.sensetime.com/pages/viewpage.action?pageId=101886408

<hr>

### usage
##### Train
```
# Modify config yaml as you want
vim ./config/resnet.yaml
cd scripts
# sh main.sh [PartitionName] [NodeNum] [ConfigFileName]
sh main.sh Test 8 resnet
```
##### Test
```
# Fill *pretrain\_model/resume\_model* in config file;
vim config/resnet.yaml  
# begin test
cd scripts
# sh eval.sh [Partition-Name] [NodeNum] [ConfigFileName] 
sh eval.sh Test 1 resnet
```
<hr>

### Version
version: 0.1

update: 2019-8-27
<hr>

### Features
##### Mix Training
```
cd scripts
sh main.sh Test 8 resnet_mix
```