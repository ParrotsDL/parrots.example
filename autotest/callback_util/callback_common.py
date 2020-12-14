"""
一些公共的数据放在这里
"""

# 公共表
comm_table = {
    '__benchmark_avg_iter_time(s)': [10000, '<', '5%'],
    '__benchmark_mem_alloc(mb)': [10000, '<', '5%'],
    '__benchmark_mem_cached(mb)': [10000, '<', '5%'],
    '__benchmark_pure_training_time(h)': [10000, '<', '5%'],
    '__benchmark_total_time(h)': [10000, '<', '5%'],
    '__benchmark_pavi_task_id': []
}

# 0: 只保存速度、显存等信息
# 1: 保存速度、显存、精度等信息
run_type_table = {
    'all': 1,
    'benchmark': 0,
    'dailytest': 1,
    'dummydata': 0,
    'weeklybenchmark': 0,
    'weeklytest': 1,
    'autoparrotsbenchmark': 0,
    'user_benchmark': 0,
    'user_all': 1,
}

value_type_table = {
    "Pattern": "max_value",
    "default": "last_value",
    "pattern_v2_5_sp": "max_value"
}

wait_time_log_no_change = 0.33  # 20 minutes for log no change(Unit is Hour)
wait_time_fork_subprocess = 0.0167  # 60 seconds for fork subprocess(Unit is Hour)
wait_time_get_slurm_jobid = 0.01 # 36 seconds for geting slurm job id(Unit is Hour)
wait_time_occur_time_limited = 0.33  # 20 minutes for occur time limited(Unit is Hour)