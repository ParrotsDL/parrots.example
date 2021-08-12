#!/usr/bin/python3
import pymysql
import time
import logging
from .db import Connection, Info

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] \
                    - %(levelname)s: %(message)s')


class DataInsert(object):
    def __init__(self):
        conn = Connection(Info.DBInfo(), Info.UserInfo())
        self.conn = conn.connection()
        self.target_table = Info.DBInfo().datatable
        self.required_keys = [
            "DataSource", "FrameName", "ModelName", "ModelDesc",
            "cluster_partition", "Storage", "CommitDate", "NumCards",
            "HostName", "IterSpeed", "FullTime", "acc1", "acc2", "acc3",
            "acc4", "AllocatedMem", "CachedMem", "TagOrBranch", "GitHash",
            "PAVIUrl", "IsParrots", "AccDesc", "ExecDate"
        ]

    def preprocess_data(self, **kwargs):
        for rk in self.required_keys:
            if rk not in kwargs:
                kwargs[rk] = None

        assert set(kwargs.keys()) == set(
            self.required_keys
        ), f'diff: {set(kwargs.keys()) - set(self.required_keys)}'

        if kwargs["CommitDate"] != '' and 'pytorch' not in kwargs["CommitDate"]:
            dt = kwargs["CommitDate"].replace("/", "-")
            timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
            kwargs["CommitDate"] = time.mktime(timeArray)
        # edpre = kwargs["ExecDate"].replace("/", "-")
        # ed = edpre.split(".", 1)[0]
        # timeArray = time.strptime(ed, "%Y-%m-%d %H:%M:%S")
        # kwargs["ExecDate"] = time.mktime(timeArray)
        return kwargs

    def insert(self, **kwargs):
        """ insert key-value into database """
        # input_data = self.preprocess_data(**kwargs)
        input_data = kwargs

        # print(input_data) #for debug
        cursor = self.conn.cursor()
        keys_str = ', '.join(list(input_data.keys()))
        tmp_str = '%s, ' * (len(input_data) - 1) + '%s'
        sql_str = f"INSERT INTO {self.target_table}({keys_str}) VALUES ({tmp_str}) "

        sql_val = list(input_data.values())
        try:
            cursor.execute(sql_str, sql_val)
            self.conn.commit()
            logging.info('insert DB ok')
            print("insert DB ok")
        except Exception as e:
            logging.error(f'insert error: {e}')
            self.conn.rollback()

    def __del__(self):
        self.conn.close()


if __name__ == '__main__':
    from fileio import load
    from db import Connection, Info

    dump_info = load(
        '/mnt/lustre/wuwenli1/workspace/benchmark_platform/auto-bench/auto_benchmark/monitor_info.json',
        file_format='json')
    print('dump_info', dump_info)
    benchmark_inserter = DataInsert()
    benchmark_inserter.insert(**dump_info)
