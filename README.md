Parrots2 examples
==

Preparation
----
Modify config yaml as you want. e.g.

./config/resnet.yaml

Run
----
*cd scripts*

sh main.sh \[PartitionName\] \[NodeNum\] \[ConfigFileName\]  e.g.

*sh main.sh Test 8 resnet*

Test
----
##### Step1
Fill *pretrain\_model/resume\_model* in config file;

##### Step2
*cd scripts*

sh eval.sh \[Partition-Name\] \[NodeNum\] \[ConfigFileName\]  e.g.

*sh eval.sh Test 1 resnet*

Features
----
##### Mix Training
*cd scripts*

*sh main.sh Test 8 resnet_mix*
