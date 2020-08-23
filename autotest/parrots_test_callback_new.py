import os
import os.path as osp
import copy
import sys
import yaml
import pavi
import time
import warnings
import re
import multiprocessing


from autoparrots.utils.fileio import dump
from autoparrots.command.entry import trace_up
from autoparrots.utils import kill_all

# 公共表
comm_table = {
    '__benchmark_avg_iter_time(s)': [10000, '<', '50%'],
    '__benchmark_mem_alloc(mb)': [10000, '<', '50%'],
    '__benchmark_mem_cached(mb)': [10000, '<', '50%'],
    '__benchmark_pure_training_time(h)': [10000, '<', '50%'],
    '__benchmark_total_time(h)': [10000, '<', '50%'],
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
    'weeklytest': 1
}


def read_log_last(path, last_line_num=5):
    if osp.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            off = -50
            while True:
                f.seek(off, 2)
                lines = f.readlines()
                if len(lines) >= last_line_num:
                    return lines
                off *= 2
    except Exception:
        return None
    return None

def _watch_for_kill_time_limited(framework, model, config):
    this_dir = osp.dirname(os.path.abspath(__file__))
    # find task.yaml
    dir_arr = this_dir.split(os.sep)
    new_dir_arr = []
    for idx, dir in enumerate(dir_arr):
        new_dir_arr.append(dir)
        if idx == 0:
            continue
        if dir_arr[idx-1] == 'tasks':
            break
    task_yaml_path = os.sep.join(new_dir_arr)
    task_yaml_path = osp.join(task_yaml_path, 'task.yaml')
    # wait for create task.yaml
    while True:
        time.sleep(1)
        if osp.exists(task_yaml_path):
            break
    tasks = yaml.load(open(task_yaml_path, 'r'), Loader=yaml.Loader)
    while True:
        time.sleep(2)
        if not isinstance(tasks, dict):
            return
        jobs = tasks['jobs']
        this_job = None
        for job in jobs:
            if (job['arg_dict']['mmaction'] == framework and
                job['arg_dict']['model'] == model):
                this_job = job
                break
        if this_job == None:
            continue
        job_pid = this_job['pid']
        job_slurm_job_id = this_job['slurm_job_id']
        job_log_path = this_job['log_path']
        # determine if a 'time limit exceeded' has occurred
        log_lines = read_log_last(job_log_path, last_line_num=10)
        is_time_limit = False
        if log_lines is not None:
            for line in log_lines:
                line = str(line, encoding="utf-8")
                if '[E] Time limit exceeded' in line:
                    kill_all(job_pid)
                    if job_slurm_job_id:
                        os.system("scancel {}".format(job_slurm_job_id))
                    is_time_limit = True
        
        if is_time_limit:
            print('Kill job {}, because of \'[E] Time limit exceeded\''.format(job_pid))
            break

def after_callback_wrapper(config, run_type):
    if run_type in config.keys():
        config = config[run_type]
    else:
        for key in run_type_table.keys():
            if key in config.keys():
                del config[key]
        if run_type_table[run_type] == 1:
            config.update(comm_table)
        else:
            config = comm_table
    if 'placeholder' in config.keys():
        del config['placeholder']

    env = os.environ.copy()
    if env.get('PAVI_TASK_ID') is not None:
        pavi_task_id = env['PAVI_TASK_ID']
    else:
        pavi_task_id = env['pavi_task_id']
    pavi_ret = dict(test_life=1)
    for k, v in config.items():
        if k == 'test_life':
            continue
        if k == '__benchmark_pavi_task_id':
            continue
        pk = 'pavi_' + k
        try:
            pv = pavi.get_scalar(pavi_task_id, k, 1)[-1]['value']
        except:
            pv = 'unknow, {} may not exist on pavi'.format(k)
            pavi_ret['test_life'] = 0
        pavi_ret[pk] = pv

    config.update(pavi_ret)
    print(yaml.dump(config))


def update_thresh_wrapper(config, framework, model_name, run_type):
    time.sleep(5)  # wait 10s for pavi scalar uploaded
    env = os.environ.copy()
    if env.get('PAVI_TASK_ID') is not None:
        pavi_task_id = env['PAVI_TASK_ID']
    else:
        pavi_task_id = env['pavi_task_id']

    root_path = trace_up('.search-run')
    root_path = root_path.replace('.search-run', 'autotest')
    configs_dir = osp.join(root_path, 'configs')
    config_path = osp.join(configs_dir, framework+'.yaml')
    full_config = copy.deepcopy(config)

    if run_type in config.keys():
        config = config[run_type]
    else:
        for key in run_type_table.keys():
            if key in config.keys():
                del config[key]
        if run_type_table[run_type] == 1:
            config.update(comm_table)
        else:
            config = comm_table
    if 'placeholder' in config.keys():
        del config['placeholder']

    if '__benchmark_pavi_task_id' not in config:
        # raise KeyError('pavi_task_id not provided')
        # make it compatible with old version
        warnings.warn('__benchmark_pavi_task_id not provided', UserWarning)
        config['__benchmark_pavi_task_id'] = []

    # TODO(shiguang): check pavi value
    update_ret = copy.deepcopy(config)
    config['test_life'] = 1
    # attr: [thresh, '>/<', '0.5%/1', val1, val2, ...]
    for k, v in config.items():
        if k == 'test_life':
            update_ret[k] = v
            continue
        if k == '__benchmark_pavi_task_id':
            pv = pavi_task_id
            update_ret[k].append(pv)
        else:
            if len(v) < 3:
                raise ValueError('{} should provid at least 3 attrs'.format(k))
            try:
                pv = pavi.get_scalar(pavi_task_id, k, 1, order_key='time')
                pv = pv[-1]['value']
            except:
                pv = 'unknow, {} may not exist on pavi'.format(k)
                config['test_life'] = 0
            update_ret[k].append(pv)
            # get value which is not string
            vaule_no_str = []
            for it in update_ret[k][3:]:
                if not isinstance(it, str):
                    vaule_no_str.append(it)
            if len(vaule_no_str) != 0:
                # update thresh
                mean_pv = 1.0 * sum(vaule_no_str) / (len(vaule_no_str))
                std_pv = float(v[2]) if not v[2].endswith('%') else float(
                    v[2][:-1]) * mean_pv * 0.01

                if v[1] == '>':
                    update_ret[k][0] = mean_pv - std_pv
                elif v[1] == '<':
                    update_ret[k][0] = mean_pv + std_pv
                else:
                    config['test_life'] = 0
                    raise KeyError('Unsupported operator key')

    full_config[run_type] = update_ret
    config = full_config

    org_config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    org_config.update({framework+'_'+model_name: config})
    dump(org_config, config_path, file_format='yaml', default_flow_style=False)

    config = config[run_type]
    print(yaml.dump(config))


def pre_callback_wrapper(config, run_type, framework, model):
    if run_type in config.keys():
        config = config[run_type]
    else:
        for key in run_type_table.keys():
            if key in config.keys():
                del config[key]
        if run_type_table[run_type] == 1:
            config.update(comm_table)
        else:
            config = comm_table
    if 'placeholder' in config.keys():
        del config['placeholder']
    config['test_life'] = 0
    print(yaml.dump(config))
    # start a thread for killing time limited
    p = multiprocessing.Process(target=_watch_for_kill_time_limited, args=(framework, model, config))
    p.setDaemon(True)
    p.start()


def collect_config(framework, model_name):
    this_dir = osp.dirname(os.path.abspath(__file__))
    configs_dir = osp.join(this_dir, 'configs')
    configs = dict()
    for f in os.listdir(configs_dir):
        config_path = osp.join(configs_dir, f)
        configs.update(
            **yaml.load(open(config_path, 'r'), Loader=yaml.Loader))
    key = framework + '_' + model_name
    return configs[key]


if __name__ == '__main__':
    assert sys.argv[4] in run_type_table.keys()
    if len(sys.argv) >= 3:
        config = collect_config(sys.argv[1], sys.argv[2])
        if sys.argv[3] == '0':
            pre_callback_wrapper(config, sys.argv[4], sys.argv[1], sys.argv[2])
        elif sys.argv[3] == '1':
            after_callback_wrapper(config, sys.argv[4])
        elif sys.argv[3] == '2':
            update_thresh_wrapper(
                config, sys.argv[1], sys.argv[2], sys.argv[4])
