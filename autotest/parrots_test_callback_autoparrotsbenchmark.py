import os
import os.path as osp
import copy
import sys
import yaml
import pavi
import time
import warnings


import psutil
try:
    from autoparrots.utils.log import LOG as logger
except:
    logger = None
from autoparrots.utils.fileio import dump
from autoparrots.command.entry import trace_up
from autoparrots.command.task import kill_task
from autoparrots.schedulers import load_taskinfo

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
    'weeklytest': 1
}

value_type_table = {
    "Pattern": "max_value",
    "default": "last_value"
}

wait_time_log_no_change = 20  # 20 minutes for log no change
wait_time_fork_subprocess = 60  # 60 seconds for fork subprocess
wait_time_get_slurm_jobid = 5 # 10 seconds for geting slurm job id
wait_time_occur_time_limited = 20  # 20 minutes for occur time limited


def read_log_last(path, last_line_num=5):
    if not osp.exists(path):
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


def get_hash(lines):
    if lines is None:
        return None
    lines_str = ' '.join(lines)
    return hash(lines_str)


def _watch_for_kill_time_limited(framework, model, config, time_limited_flag='[E] Time limit exceeded'):
    time.sleep(60)
    # wait for job_pid, job_log_path
    job_pid = None
    slurm_job_id = None
    job_log_path = None
    workdir = None
    name = None
    job_wait_to_run_time_thresh = 1  # wait one hour
    start_time = time.time()
    while True:
        time.sleep(30)
        interval_time = time.time() - start_time
        if ((not job_pid or
             not job_log_path or
             not workdir or
             not name or
             not slurm_job_id) and
                interval_time >= job_wait_to_run_time_thresh * 60 * 60):
            break

        # get job pid
        try:
            job_pid = int(os.environ['pid'])
        except Exception:
            job_pid = None
        # get job wait_to_run_time_thresh
        try:
            job_wait_to_run_time_thresh = int(
                os.environ['wait_to_run_time_thresh'])
        except Exception:
            job_wait_to_run_time_thresh = 1
        # get job log path
        job_log_path = os.environ['log_path']
        # get job workdir
        try:
            workdir = os.environ['workdir']
            workdir = os.sep.join(workdir.split(os.sep)[:-2])
        except:
            workdir = None
        # get job name
        name = os.environ['name']
        # get slurm_job_id
        if workdir and name:
            info = load_taskinfo(workdir)
            job_names = list(
                filter(lambda j: j['name'] in [name], info['jobs']))
            if len(job_names) > 0:
                job_info = job_names[0]
                try:
                    slurm_job_id = int(job_info['slurm_job_id'])
                except Exception:
                    slurm_job_id = None
        _, status = get_slurm_job_id()
        if job_pid and job_log_path and workdir and name and slurm_job_id and status and status == 'R':
            break
        # break if job_pid is die.
        if job_pid and (not psutil.pid_exists(job_pid)):
            break

    # monitor log
    last_lines_hash = None
    last_lines_hash_start_time = time.time()
    time_limited_start_time = None
    while True:
        if (not job_pid or
            not job_log_path or
            not workdir or
            not name or
                not slurm_job_id):
            break
        time.sleep(60)
        is_time_limit = False
        # get last some lines
        log_lines = read_log_last(job_log_path, last_line_num=10)
        if log_lines is not None:
            log_lines = [str(line, encoding="utf-8") for line in log_lines]
        # get log hash
        lines_hash = get_hash(log_lines)
        # monitor whether the log has not changed over time (kill all process if not change for a long time)
        if lines_hash == last_lines_hash:
            if time.time() - last_lines_hash_start_time >= wait_time_log_no_change * 60:
                kill_task(workdir, [name])
                is_time_limit = True
                if logger:
                    logger.error("Job({})[pid: {}, slurm: {}] is killed because the log has not changed for {} minutes.".format(
                        name, job_pid, slurm_job_id, wait_time_log_no_change))
        else:
            last_lines_hash = lines_hash
            last_lines_hash_start_time = time.time()
        # monitor whether a 'time limit exceeded' has occurred
        if (log_lines is not None) and (not is_time_limit):
            is_time_limit_occur = False
            for line in log_lines:
                if time_limited_flag in line:
                    is_time_limit_occur = True
                    break
            if is_time_limit_occur and time_limited_start_time is not None:
                if time.time() - time_limited_start_time >= wait_time_occur_time_limited * 60:
                    kill_task(workdir, [name])
                    is_time_limit = True
                    if logger:
                        logger.error("Job({})[pid: {}, slurm: {}] is killed because the log occurs '{}' for {} minutes.".format(
                            name, job_pid, slurm_job_id, time_limited_flag, wait_time_occur_time_limited))
            else:
                time_limited_start_time = time.time()

        # break if occur '[E] Time limit exceeded'
        if is_time_limit:
            break
        # break if job_pid is die.
        if not psutil.pid_exists(job_pid):
            break

    if logger:
        logger.info("Job({})[pid: {}, slurm: {}] The child process monitoring the log has exited, \
                      with [job_pid: {}, job_log_path: {}, workdir: {}, name: {}, slurm_job_id: {}]".format(
            name, job_pid, slurm_job_id, job_pid, job_log_path, workdir, name, slurm_job_id))


def after_callback_wrapper(config, value_type, run_type):
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

        if value_type == "max_value":
            try:
                if v[1] == '>':
                    pv = sorted(pavi.get_scalar(pavi_task_id, k, 10),
                                key=lambda x: x.__getitem__('value'))[-1]['value']
                    pavi_ret[pk] = pv
                else:
                    pv = sorted(pavi.get_scalar(pavi_task_id, k, 10), key=lambda x: x.__getitem__(
                        'value'), reverse=True)[-1]['value']
                    pavi_ret[pk] = pv
            except:
                pv = 'unknow, {} may not exist on pavi'.format(k)
                pavi_ret['test_life'] = 0

        elif value_type == "last_value":
            try:
                pv = pavi.get_scalar(pavi_task_id, k, 1)[-1]['value']
                pavi_ret[pk] = pv
            except:
                pv = 'unknow, {} may not exist on pavi'.format(k)
                pavi_ret['test_life'] = 0
        else:
            print("Please set 'max' or 'last' for the type of value.")

    config.update(pavi_ret)
    print(yaml.dump(config))


def get_benchmark_value(config, framework, model_name, value_type, run_type):
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

        if value_type == "max_value":
            try:
                if v[1] == '>':
                    pv = sorted(pavi.get_scalar(pavi_task_id, k, 10),
                                key=lambda x: x.__getitem__('value'))[-1]['value']
                    pavi_ret[pk] = pv
                else:
                    pv = sorted(pavi.get_scalar(pavi_task_id, k, 10), key=lambda x: x.__getitem__(
                        'value'), reverse=True)[-1]['value']
                    pavi_ret[pk] = pv
            except:
                pv = 'unknow, {} may not exist on pavi'.format(k)
                pavi_ret['test_life'] = 0

        elif value_type == "last_value":
            try:
                pv = pavi.get_scalar(pavi_task_id, k, 1)[-1]['value']
                pavi_ret[pk] = pv
            except:
                pv = 'unknow, {} may not exist on pavi'.format(k)
                pavi_ret['test_life'] = 0
        else:
            print("Please set 'max' or 'last' for the type of value.")

    # TODO(zhouhanyu): get pavi benchmark value
    root_path = trace_up('.search-run')
    root_path = root_path.replace('.search-run', 'autotest')
    configs_dir = osp.join(root_path, 'benchmark')
    config_path = osp.join(configs_dir, framework+'.yaml')
    if os.path.exists(config_path):
        org_config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        org_config.update({framework+'_'+ model_name: pavi_ret})
    else:
        org_config = {framework+'_'+ model_name: pavi_ret}
    dump(org_config, config_path, file_format='yaml', default_flow_style=False)


def update_thresh_wrapper(config, framework, model_name, value_type, run_type):
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
            if value_type == "max_value":
                try:
                    if v[1] == '>':
                        pv = sorted(pavi.get_scalar(
                            pavi_task_id, k, 10, order_key='time'), key=lambda x: x.__getitem__('value'))
                        pv = pv[-1]['value']
                    else:
                        pv = sorted(pavi.get_scalar(pavi_task_id, k, 10, order_key='time'),
                                    key=lambda x: x.__getitem__('value'), reverse=True)
                        pv = pv[-1]['value']
                except:
                    pv = 'unknow, {} may not exist on pavi'.format(k)
                    config['test_life'] = 0
            elif value_type == "last_value":
                try:
                    pv = pavi.get_scalar(pavi_task_id, k, 1, order_key='time')
                    pv = pv[-1]['value']
                except:
                    pv = 'unknow, {} may not exist on pavi'.format(k)
                    config['test_life'] = 0
            else:
                print("Please set 'max' or 'last' for the type of value.")

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

def get_slurm_job_id():
    """
    get slurm_job_id by squeue
    """
    work_dir = os.environ['run_path']
    command = os.environ['command']
    # find srun_args
    partition = None
    srun_args = os.environ.get('SRUN_ARGS', None)
    if srun_args != None:
        try:
            srun_args = srun_args.split(' ')
            for idx, args in enumerate(srun_args):
                if args == '-p':
                    partition = srun_args[idx+1]
                    break
        except Exception as e:
            logger.warn("can't get partition from srun_args")
    else:
        try:
            command_arr = command.split(' ')
            for idx, val in enumerate(command_arr):
                if val.endswith('train.sh'):
                    partition = command_arr[idx+1]
                    break
        except Exception as e:
            logger.warn("can't get partition from command")

    if not partition:
        squeue_command = 'squeue -o "%.50i %.50j %.20u %t %D %N"'
    else:
        squeue_command = 'squeue -o "%.50i %.50j %.20u %t %D %N" -p {}'.format(partition)

    try:
        slurm_job_id = None
        status = None
        ret = os.popen(squeue_command).read()
        ret = ret.strip('\n').split('\n')
        ret = ret[1:]
        ret = [item.strip(' ').split(' ') for item in ret]
        new_ret = []
        for idx, item in enumerate(ret):
            tmp_ret = []
            for subidx, subitem in enumerate(item):
                if subitem != '':
                    tmp_ret.append(subitem)
            new_ret.append(tmp_ret)
        task_infos = new_ret

        for task_info in task_infos:
            jobid = task_info[0]
            ret = os.popen("scontrol show job {}".format(jobid)).read()
            ret = ret.split('\n')
            for item in ret:
                if 'WorkDir' in item:
                    workdir_tmp = item.split('=')[-1]
                    if workdir_tmp == work_dir:
                        slurm_job_id = jobid
                        status = task_info[3]
                        break
        if not slurm_job_id:
            logger.warn("can't get slurm from squeue")
            return None, None
        return slurm_job_id, status
    except Exception as e:
        logger.warn("can't get slurm from squeue")
        return None, None

def pre_callback_wrapper(config, run_type, framework, model, is_monitor_log=True):
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
    config['__benchmark_total_time(h)'] = 0.2

    # get slurm job id
    slurm_job_id = ''
    status = ''
    start_time = time.time()
    while True:
        interval_time = time.time() - start_time
        if interval_time >= wait_time_get_slurm_jobid:
            break
        slurm_job_id, status = get_slurm_job_id()
        if slurm_job_id:
            break
    config['slurm_job_id'] = slurm_job_id
    config['slurm_job_status'] = status

    print(yaml.dump(config))
    if is_monitor_log:
        # start a process for killing time limited
        pid = -1
        start_time = time.time()
        while pid < 0:
            interval_time = time.time() - start_time
            if interval_time >= wait_time_fork_subprocess:
                break
            pid = os.fork()
        if pid == 0:
            _watch_for_kill_time_limited(framework, model, config)


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
        if sys.argv[1] in value_type_table.keys():
            value_type = value_type_table[sys.argv[1]]
        else:
            # default: read last_value
            value_type = value_type_table["default"]
        if sys.argv[3] == '0':
            pre_callback_wrapper(config, sys.argv[4], sys.argv[1], sys.argv[2])
        elif sys.argv[3] == '1':
            after_callback_wrapper(config, value_type, sys.argv[4])
        elif sys.argv[3] == '2':
            update_thresh_wrapper(
                config, sys.argv[1], sys.argv[2], value_type, sys.argv[4])
        elif sys.argv[3] == '3':
            get_benchmark_value(config, sys.argv[1], sys.argv[2], value_type, sys.argv[4])
