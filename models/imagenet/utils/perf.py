from time import time
from datetime import datetime
import numpy as np
import json
import os


class PerfRecorder(object):
    def __init__(self,
                 path,
                 batchsize=None,
                 num_gpus=None,
                 rank=None,
                 trim_head=None) -> None:
        super().__init__()
        self.path = path
        self.batchsize = batchsize
        self.num_gpus = num_gpus
        self.rank = rank
        self.trim_head = trim_head

        self.record = {}
        self.recording = {}

    def set_start_timer(self, timer_name):
        if timer_name not in self.record.keys():
            self.record[timer_name] = [time()]
        elif self.recording[timer_name]:
            self.record[timer_name][-1] = [time()]
        else:
            self.record[timer_name].append(time())
        self.recording[timer_name] = True

    def set_end_timer(self, timer_name):
        assert timer_name in self.record.keys(), \
            'no start_timer {} found'.format(timer_name)
        if len(self.record[timer_name]) > 0:
            self.record[timer_name][-1] = time() - self.record[timer_name][-1]
        self.recording[timer_name] = False

    def gen_perf_results(self):
        res = {}
        for k in self.record.keys():
            if self.recording[k]:
                self.record[k].pop()
            if len(self.record[k]) > 0:
                res[k] = {
                    'mean': np.around(np.mean(self.record[k])),
                    'var': np.around(np.var(self.record[k]))
                }

        # fps
        if self.batchsize is not None and self.num_gpus is not None \
                and 'iter' in self.record.keys():
            fps = [
                self.batchsize * self.num_gpus / iter_time
                for iter_time in self.record['iter']
            ]
            res['fps'] = {
                'mean': np.around(np.mean(fps), 3),
                'var': np.around(np.var(fps), 3)
            }

        # print(res)
        timestamp = str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        perf_record_file = os.path.join(
            self.path, 'card-' + str(self.rank) + '-perf-record-' + timestamp)
        with open(perf_record_file, 'w') as f:
            f.write(json.dumps(res))
