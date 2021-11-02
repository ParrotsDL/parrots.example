# 问答Bert-Base-Cased的Pytorch训练
  
## 训练准备

### 环境和数据准备
    export CNCL_MLULINK_TIMEOUT_SECS=-1 (turn off the timeout of ddp communication)
    pytorch 1.3.0
    python 3.6
    pip install transformers==3.5.1
    dataset: export SQUAD_DIR
    pretrain_weight: 将预训练权重放在~/.cache/torch/下 

### MLU scratch训练
    bash scratch_bert_base_cased_4mlu.sh
