# transformer

## for 1424

```shell
source  /mnt/lustre/share/platform/env/pat_latest
export DATASET_PATH=/mnt/lustre/share/nlp/corpora/
sh train_gpu.sh partition_name cards_num --device gpu
```

## for 1984

```shell
source pat_latest
export DATASET_PATH=/mnt/lustre/share_data/jiangyongjiu/datasets/corpora/
sh train_gpu.sh partition_name cards_num --device gpu
```

## for 200 pytorch1.6

```shell
source /mnt/lustre/share/platform/env/camb_pytorch1.6
export DATASET_PATH=/mnt/lustre/share/datasets/nlp/corpora/
sh train_mlu.sh partition_name cards_num
```

## for 200 parrots

```shell
source /mnt/lustre/share/platform/env/pat2.0_dev_gcc5.4
export PYTHONPATH=/mysenseparrotspath:$PYTHONPATH
export DATASET_PATH=/mnt/lustre/share/datasets/nlp/corpora/
// train model
sh train_mlu.sh partition_name cards_num

// test model
export MODEL_PATH=models/model_epoch_$num.pth
sh test_mlu.sh partiton_name
```

