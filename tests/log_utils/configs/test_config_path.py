import os

CONFIG_DIR = '.parrots'
CONFIG_PATH = os.path.join(os.environ['HOME'], CONFIG_DIR)

# 删除 config 目录
os.system("rm -rf " + CONFIG_PATH)
assert os.path.isdir(CONFIG_PATH) is False

import torch

# 验证 config 目录生成
if os.path.isdir(CONFIG_PATH):
    print("Test Successfully!")
else:
    print("Test Failed!")
