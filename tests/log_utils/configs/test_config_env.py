import os

# 测试已设置环境变量不会被覆盖
os.environ['PYTORCH_VERSION'] = '1.8'

# 测试加载parrots环境变量
import torch    # noqa E402

pytorch_version = os.getenv('PYTORCH_VERSION')
assert pytorch_version == '1.8'
# 测试加载环境变量默认值
parrots_opbenchmark = os.getenv('PARROTS_OPBENCHMARK')
assert parrots_opbenchmark == 'OFF'
parrots_timeline_file = os.getenv('PARROTS_TIMELINE_FILENAME')
assert parrots_timeline_file == 'timeline.json.gz'

# 测试默认生成环境变量文件存在
CONFIG_DIR = '.parrots'
CONFIG_PATH = os.path.join(os.environ['HOME'], CONFIG_DIR)
CONFIG_FILENAME = 'config.yaml'
CONFIG_FILE = os.path.join(CONFIG_PATH, CONFIG_FILENAME)
assert os.path.isfile(CONFIG_FILE) is True
