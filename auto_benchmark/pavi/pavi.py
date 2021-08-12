import os
import sys
from auto_benchmark.utils import DataInsert, getinfo
import logging
import torch.distributed as dist
import configparser
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] \
                    - %(levelname)s: %(message)s')

monitor={}

class SummaryWriter:
    def __init__(self, *args, **kwargs):
        logging.info("using fake pavi.....")
        self.monitor_info = getinfo.get_benchmark_info()

        benchmark_inserter = DataInsert()
        logging.info(f"{self.monitor_info}")
        self.monitor_info = benchmark_inserter.preprocess_data(
            **self.monitor_info)
        # time.time()
        detail_info = os.environ.get('DETAIL_INFO', None)

        # detail_info = 'example*1*resnet18'
        if detail_info:
            self.detail_infos = detail_info.split('*')
        else:
            logging.info("!!! DETAIL_INFO of Model  is missing, \
                Please check the env, so exit the process.")
            exit(0)

        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read(
            os.path.join(
                os.path.abspath(__file__).rsplit('/', 2)[0], 'db.ini'))
        self.datasource = config['db_table_info']['datatable']

        # get model info
        self.monitor_info['FrameName'] = self.detail_infos[0]
        self.monitor_info['NumCards'] = self.detail_infos[1]
        self.monitor_info[
            'ModelName'] = self.detail_infos[2] + "*" + os.environ.get(
                'BENCHMARK_FS', 'default')
        self.monitor_info['ModelDesc'] = " "
        self.monitor_info['cluster_partition'] = os.environ.get(
            'Partition', "")
        self.monitor_info['DataSource'] = self.datasource

    def add_scalar(self, tag, value, iteration=0):
        global monitor
        if tag == "__benchmark_avg_iter_time(s)" and dist.get_rank() == 0:

            if value > 0 and dist.get_rank() == 0:
                self.monitor_info['IterSpeed'] = value
            else:
                logging.info(
                    f'!!! {tag} is not a normal value, so exit the process.')
                exit(0)
        elif tag == "__benchmark_pure_training_time(h)" and dist.get_rank(
        ) == 0:
            if value > 0:
                self.monitor_info['FullTime'] = value
            else:
                logging.info(
                    f'!!! {tag} is not a normal value, so exit the process.')
                exit(0)
        else:
            pass
        monitor = self.monitor_info

    def func(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return self.func


def upload():
    #for debug
    print(monitor)
    if os.environ.get('DETAIL_INFO', None):
        if dist.get_rank(
        ) == 0 and monitor['IterSpeed'] > 0 and monitor['FullTime'] > 0:
            benchmark_inserter = DataInsert()
            benchmark_inserter.insert(**monitor)
        else:
            logging.info(f'{monitor} failed to write to db!')
            print(f'{monitor} failed to write to db!')
    else:
        pass


import atexit
atexit.register(upload)
