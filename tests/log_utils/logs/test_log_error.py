import os
import glob
import datetime


PARROTS_DIR = '.parrots'
LOG_DIR = 'logs'
LOG_DIR_PATH = os.path.join(os.path.join(os.environ['HOME'], PARROTS_DIR),
                            LOG_DIR)
LOG_DATE = datetime.datetime.now()
LOG_DATE_DIR = str(LOG_DATE.year) + "-" + str(LOG_DATE.month).zfill(2) + \
    "-" + str(LOG_DATE.day).zfill(2)
# log文件路径
LOGFILE_DIR = os.path.join(LOG_DIR_PATH, LOG_DATE_DIR)
LOGFILENAMES = glob.glob(LOGFILE_DIR + "/.log*")
# 获取logs数量
LOGFILES_NUM = len(LOGFILENAMES)


def test_log_error():
    import torch
    import parrots
    a = torch.ones((2, 2))
    b = torch.ones((3, 3))

    # an error
    try:
        a + b
    except ValueError as e:
        parrots.log_utils.log_info('get an ValueError:\n {}'.format(str(e)))

    # 测试报错文件路径存在
    assert os.path.isdir(LOGFILE_DIR)
    # 测试生成报错log, log数量加 1
    LOGFILENAMES = glob.glob(LOGFILE_DIR + "/.log*")
    assert len(LOGFILENAMES) == LOGFILES_NUM + 1
    print('test successfully!')


if __name__ == '__main__':
    test_log_error()
