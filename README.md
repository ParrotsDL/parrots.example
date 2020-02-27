## Parrots2 examples
Image Classification Training in Parrots and Pytorch.

Get detailed documentation in http://pape.parrots.sensetime.com/example_imagenet.html

<hr>

### usage
##### Train
```
# cd imagenet dir
cd models/imagenet/
# Modify config yaml as you want
vim configs/resnet.yaml
# start training
# sh train.sh [ConfigFileName] [JobName] [PartitionName] [NodeNum]
sh train.sh configs/resnet.yaml resnet Test 8
```
##### Test
```
# cd imagenet dir
cd models/imagenet/
# Fill *pretrain\_model/resume\_model* in config file
vim configs/resnet.yaml  
# start testing
# sh test.sh [ConfigFileName] [JobName] [PartitionName] [NodeNum]
sh test.sh configs/resnet.yaml test Test 1
```
<hr>

### Version
version: 0.2

update: 2020-02-27
<hr>

### Features
##### Mix Training
```
sh train.sh configs/resnet50_mix.yaml resnet Test 8
```
##### Training with SyncBN
```
sh train.sh configs/resnet50_syncbn.yaml resnet Test 8
```
