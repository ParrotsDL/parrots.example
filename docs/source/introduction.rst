概述
====
Parrots.example 是一个同时支持Parrots 和 Pytorch 环境下训练的分类模型框架，同时支持多个经典模型的单精度和混合精度训练。

一、总体架构
------------------
1、模型
""""""""""""""""""
+------------------------+--------------------------------------------------------------------+
| 模块名                 | 功能                                                               |
+========================+====================================================================+
| inception              | v1, v2, v3, v4                                                     |
+------------------------+--------------------------------------------------------------------+
| resnet                 | resnet18, resnet34,resnet50, resnet101, resnet152                  |
+------------------------+--------------------------------------------------------------------+
| resnet_v2              | resnet18_v2, resnet34_v2,resnet50_v2, resnet101_v2, resnet200_v2   |
+------------------------+--------------------------------------------------------------------+
| inception_resnet       | v1, v2                                                             |
+------------------------+--------------------------------------------------------------------+
| dpn                    | dpn92                                                              |
+------------------------+--------------------------------------------------------------------+
| mobilenet              | v1, v2                                                             |
+------------------------+--------------------------------------------------------------------+
| senet                  | v1, v2                                                             |
+------------------------+--------------------------------------------------------------------+
| shufflenet             | v1, v2                                                             |
+------------------------+--------------------------------------------------------------------+
| vgg                    | vgg16                                                              |
+------------------------+--------------------------------------------------------------------+

2、功能支持
""""""""""""""""""
1）数据读取方式
******************
  目前支持的数据读取方式：pillow / opencv. 使用opencv读取数据之后，需要转换成PIL图片的形式，方便后面进行图片的transfrom操作

2）学习率调整方式
******************
  目前支持的学习率调整方式如下：

  - MultiStepLR
  - EpochStepLR
  - LinearLR
  - CosineLR

3）可视化及保存 snapshot 方式
******************
  通过配置文件里面的 ``monitor`` 字段来选择，目前支持的可视化方式如下：

  - pavi
  - tensorboard

  通过配置文件里面的 ``saver`` 字段来选择，目前支持的可视化方式如下：

  - pavi
  - lustre


4）混合精度训练
******************
  使用 FP16 来进行计算可以降低内存消耗，减少 Training 和 Inference 的计算时间。但是 FP16 的数值范围小于 FP32，部分网络训练存在溢出和精度损失的问题。

  该训练方案仅在支持 pape 环境下可进行，pape中使用了 loss scaling 策略和 FP32 master copy of weights 策略来解决上述问题，具体原理可以点击链接查看。


5）模型转换
******************
  Parrots.example 支持 Parrots 和 Pytorch 之间的模型转换，是以 numpy 作为中间结果来进行转换。转换代码在 ``./tools/model_transform.py``. 以将 Pytorch 
  模型转换为 Parrots 模型为例，代码如下：

.. code-block:: bash

  # PyTorch model -> numpy model
  # 先激活 Pytorch 环境
  source pt0.4v1
  cd tools
  python model_transform.py -i torch_state_dict.pkl -o np_state_dict.pkl -to_np
  # 退出环境
  source pat_deactivate
  # numpy model -> Parrots model
  # 激活 Parrots 环境
  source pat_latest
  python model_transfrom.py -i np_state_dict.pkl -o torch_state_dict.pkl -to_tensor


6）模型并行训练
******************
  在多卡并行的场景下，我们一般都是使用数据并行策略，但对于一些参数运算量特别大的场景，模型并行是一种很好的方法。目前支持的是主干网络数据并行，自定义的 某些层使用模型并行，可以解决类别数过多，单张卡放不下的问题。该部分代码尚未开源。

二、使用方式
""""""""""""""""""
在parrots和pytorch环境下均可直接使用，集群环境可看这里：`集群环境 <https://confluence.sensetime.com/pages/viewpage.action?pageId=82126258>`_

.. code-block:: bash

  # 准备代码
  git clone git@gitlab.bj.sensetime.com:platform/ParrotsDL/parrots.example.git
  cd parrots.example
  # 训练
  vim configs/resnet.yaml    # 修改相关配置文件的相关参数
  cd scripts
  sh main.sh Test 8 resnet  # sh main.sh [PartitionName] [NodeNum] [ConfigFileName]
  # 测试
  vim configs/resnet.yaml  # 重点修改 pretrain_model
  cd scripts
  sh eval.sh Test 1 resnet  # sh eval.sh [PartitionName] [NodeNum] [ConfigFileName]


三、精度结果
""""""""""""""""""
parrots.example 可直接在 Parrots 和 Pytorch 环境下进行训练，支持多个模型，精度结果见 `Model Zoo <https://confluence.sensetime.com/display/Parrots/Model+Zoo>`_



