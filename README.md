## Parrots2 examples for k8s
Classification models for testing parrots2 on kubernetes
<hr>

### Use k8s or raw mpirun to train models
Get detailed information on https://confluence.sensetime.com/pages/viewpage.action?pageId=117756142
##### k8s
Modify training parameters like num of GPUs, num of servers, epochs in `./scripts/run_on_k8s.yaml`
```
cd scripts/
./run_on_k8s.yaml
```

##### raw mpirun
Modify training parameters like num of GPUs, num of servers, epochs in `./scripts/run_raw.yaml`
```
cd scripts/
./run_raw.yaml
```

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