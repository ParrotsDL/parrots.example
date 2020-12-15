#!/usr/bin/python3
import pymysql
import time
import sys
try:
    from autoparrots.utils.log import LOG as logger
except:
    logger = None

class DataInseter(object):
    def __init__(self,
                 host='sh.paas.sensetime.com',
                 user='fsneogke',
                 password='xdcsveww',
                 db='parrots',
                 port=38152,
                 local_infile=1):
        self.db = pymysql.connect(host=host, user=user, password=password,
                                  db=db, port=port, local_infile=local_infile)
        self.required_keys = ["DataSource", "FrameName", "ModelName", "ModelDesc", "cluster_partition",
                              "Storage", "CommitDate", "NumCards", "HostName", "IterSpeed",
                              "FullTime", "acc1", "acc2", "acc3", "acc4", "AllocatedMem",
                              "CachedMem", "TagOrBranch", "GitHash", "PAVIUrl", "IsParrots",
                              "AccDesc", "ExecDate"
                             ]

    def preprocess_data(self, **kwargs):
        for rk in self.required_keys:
            if rk not in kwargs:
                kwargs[rk] = None
        assert set(kwargs.keys()) == set(
            self.required_keys), f'diff: {set(kwargs.keys()) - set(self.required_keys)}'

        if kwargs["CommitDate"] != '':
            dt = kwargs["CommitDate"].replace("/", "-")
            timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
            kwargs["CommitDate"] = time.mktime(timeArray)
        
        edpre = kwargs["ExecDate"].replace("/", "-")
        ed = edpre.split(".",1)[0]
        timeArray = time.strptime(ed, "%Y-%m-%d %H:%M:%S")
        kwargs["ExecDate"] = time.mktime(timeArray)
        
        return kwargs

    def insert(self, **kwargs):
        """ insert key-value into database """

        input_data = self.preprocess_data(**kwargs)
        cursor = self.db.cursor()
        keys_str = ', '.join(list(input_data.keys()))
        tmp_str = '%s, ' * (len(input_data)-1) + '%s'
        sql_str = f"INSERT INTO modelmonitor({keys_str}) VALUES ({tmp_str}) "
        sql_val = list(input_data.values())
        try:
            cursor.execute(sql_str, sql_val)
            self.db.commit()
            if logger:
                logger.info('insert ok')
        except Exception as e:
            if logger:
                logger.error(f'insert error: {e}')
            self.db.rollback()

    def __del__(self):
        self.db.close()

if __name__ == '__main__':
    from autoparrots.utils.fileio import load
    dump_info = load('monitor_info.json', file_format='json')
    benchmark_inserter = DataInseter()
    benchmark_inserter.insert(**dump_info)





