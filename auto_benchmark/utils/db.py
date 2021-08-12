import pymysql
import time
import logging
import configparser
import os


class Info(object):
    class DBInfo:
        def __init__(self, host=None, port=None, db=None, datatable=None):
            config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            config.read(
                os.path.join(
                    os.path.abspath(__file__).rsplit('/', 2)[0], 'db.ini'))
            self.host = config['db_cfg_info']['host']
            self.port = int(config['db_cfg_info']['port'])
            self.db = config['db_table_info']['db']
            self.datatable=config['db_table_info']['datatable']
            self.sql = config['sql_commmod']['sql']

    class UserInfo(object):
        def __init__(self, name=None, password=None):
            config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            config.read(
                os.path.join(
                    os.path.abspath(__file__).rsplit('/', 2)[0], 'db.ini'))
            self.name = config['user_info']['name']
            self.password = config['user_info']['password']


class Connection(object):
    def __init__(self, dbinfo, userinfo, charset='utf-8'):
        self.dbinfo = dbinfo
        self.userinfo = userinfo
        if not hasattr(self, 'conn'):
            self.conn = pymysql.connect(host=self.dbinfo.host,
                                        port=self.dbinfo.port,
                                        user=self.userinfo.name,
                                        db=self.dbinfo.db,
                                        password=self.userinfo.password,
                                        local_infile=1)
            logging.info('connection initialized successfully.')

    def __del__(self):
        # try:
        #     self.conn.ping()  # 采用连接对象的ping()函数检测连接状态，出现异常表示已经关了，否则关闭
        #     self.conn.close()
        # except:
        #     logging.info('connection has closed.')
        logging.info('connection closed successfully.')

    def connection(self):
        return self.conn