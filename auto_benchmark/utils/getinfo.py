import os
import time
import logging
import pymysql
from .db import Connection, Info


def get_benchmark_info():
    # DataSource: dailybuild, weeklybuild, release_benchmark, weekly_benchmark
    DataSource = ''
    FrameName = ''
    ModelName = ''
    ModelDesc = ''  #srun
    partition = ''
    IterSpeed = 0
    FullTime = 0
    NumCards = 0

    ExecDate = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    HostName = os.uname().nodename
    # CommitDate ,TagOrBranch , GitHash,IsParrots,
    IsParrots = False
    try:
        import torch
        if torch.__version__ == 'parrots':
            import parrots
            IsParrots = True
            CommitDate = parrots.info.git_latest_commit_date
            TagOrBranch = parrots.info.git_tag_or_branch
            GitHash = parrots.version.git_hash
        else:
            CommitDate = 'pytorch'
            TagOrBranch = 'pytorch'
            GitHash = 'pytorch'
    except ModuleNotFoundError:
        CommitDate, TagOrBranch, GitHash = '', '', ''

    Storage = ''
    AccDesc = ''
    PAVIUrl = ''
    acc_list = [None] * 4
    AllocatedMem = 0
    CachedMem = 0

    monitor_info = dict(IsParrots=IsParrots,
                        DataSource=DataSource,
                        Storage=Storage,
                        CommitDate=CommitDate,
                        HostName=HostName,
                        AllocatedMem=AllocatedMem,
                        CachedMem=CachedMem,
                        TagOrBranch=TagOrBranch,
                        GitHash=GitHash,
                        PAVIUrl=PAVIUrl,
                        ExecDate=ExecDate,
                        acc1=acc_list[0],
                        acc2=acc_list[1],
                        acc3=acc_list[2],
                        acc4=acc_list[3],
                        AccDesc=AccDesc)
    return monitor_info
