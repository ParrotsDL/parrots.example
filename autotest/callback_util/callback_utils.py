"""
一些通用的方法放在这里
"""
import os
import pavi
import time
import copy

from . import callback_common, logger

def get_slurm_job_id():
    """
    get slurm_job_id by squeue
    return: slurm_job_id, partition, status
    """
    work_dir = os.environ['run_path']
    command = os.environ['command']
    name = os.environ['name']
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
            logger.warn(
                "Job({}): can't get partition from srun_args".format(name))
    else:
        try:
            command_arr = command.split(' ')
            for idx, val in enumerate(command_arr):
                if val.endswith('train.sh'):
                    partition = command_arr[idx+1]
                    break
        except Exception as e:
            logger.warn(
                "Job({}): can't get partition from command".format(name))

    if not partition:
        squeue_command = 'squeue -o "%.50i %.50j %.20u %t %D %N"'
    else:
        squeue_command = 'squeue -o "%.50i %.50j %.20u %t %D %N" -p {}'.format(
            partition)

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
            return None, partition, None
        return slurm_job_id, partition, status
    except Exception as e:
        return None, partition, None



def get_monitor_info(config, run_type):
    # DataSource: dailybuild, weeklybuild, release_benchmark, weekly_benchmark
    DataSource = run_type
    # Partition
    _, partition, _ = get_slurm_job_id()

    # NumCards, TODO: Storage, parse from command
    NumCards = 0
    Storage = ''
    FrameName = ''
    ModelName = ''
    command_arr = os.environ['command'].split(' ')
    for idx, val in enumerate(command_arr):
        if val.endswith('train.sh'):
            FrameName = val.split('/')[-2]
            NumCards = command_arr[idx+2]
            ModelName = command_arr[idx+3]
            break
    if NumCards == 0:
        logger.error('Can not get cards num from command')
    # CommitDate and FrameName
    IsParrots = False
    try:
        import parrots
        import torch
        if torch.__version__ == 'parrots':
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

    IterSpeed = config.pop('pavi___benchmark_avg_iter_time(s)')
    FullTime = config.pop('pavi___benchmark_total_time(h)')
    AllocatedMem = config.pop('pavi___benchmark_mem_alloc(mb)')
    CachedMem = config.pop('pavi___benchmark_mem_cached(mb)')
    config.pop('pavi___benchmark_pure_training_time(h)')
    # transform acc
    acc_list = [None] * 4
    AccDesc = []
    idx = 0
    for k, v in config.items():
        if k.startswith('pavi_'):
            accmap = f'acc{idx+1}: {k}'
            AccDesc.append(accmap)
            acc_list[idx] = v
            idx += 1
    AccDesc = ', '.join(AccDesc)
    PAVIUrl = None
    try:
        # for pavi 2.0
        PAVIUrl=pavi.get_training_url(os.environ['pavi_training_id'])
    except Exception:
        pass

    try:
        # for pavi 1.0
        PAVIUrl = '{}/#/task/{}'.format(
            pavi.Config.PAVI_SERVER.value, os.environ['pavi_task_id'])
    except Exception:
        pass

    monitor_info = dict(
        IsParrots=IsParrots,
        DataSource=DataSource,
        FrameName=FrameName,
        ModelName=ModelName,
        ModelDesc=os.environ['command'],
        cluster_partition=partition,
        Storage=Storage,
        CommitDate=CommitDate,
        NumCards=NumCards,
        HostName=os.environ['host_ip'],
        IterSpeed=IterSpeed,
        FullTime=FullTime,
        AllocatedMem=AllocatedMem,
        CachedMem=CachedMem,
        TagOrBranch=TagOrBranch,
        GitHash=GitHash,
        PAVIUrl=PAVIUrl,
        ExecDate=os.environ['start_time'],
        acc1=acc_list[0],
        acc2=acc_list[1],
        acc3=acc_list[2],
        acc4=acc_list[3],
        AccDesc=AccDesc
    )
    return monitor_info

def get_scalar_from_pavi(config, value_type, run_type, is_update_yaml=False):
    if run_type in config.keys():
        config = config[run_type]
    else:
        for key in callback_common.run_type_table.keys():
            if key in config.keys():
                del config[key]
        if callback_common.run_type_table[run_type] == 1:
            config.update(callback_common.comm_table)
        else:
            config = callback_common.comm_table
    if 'placeholder' in config.keys():
        del config['placeholder']

    if is_update_yaml:
        if '__benchmark_pavi_task_id' not in config:
            # raise KeyError('pavi_task_id not provided')
            # make it compatible with old version
            warnings.warn('__benchmark_pavi_task_id not provided', UserWarning)
            config['__benchmark_pavi_task_id'] = []

    env = os.environ.copy()
    if env.get('PAVI_TASK_ID') is not None:
        pavi_task_id = env['PAVI_TASK_ID']
    else:
        pavi_task_id = env['pavi_task_id']

    if is_update_yaml:
        update_ret = copy.deepcopy(config)
        config['test_life'] = 1
    else:
        pavi_ret = dict(test_life=1)
    for k, v in config.items():
        if k == 'test_life':
            if is_update_yaml:
                update_ret[k] = v
            continue
        if k == '__benchmark_pavi_task_id':
            if is_update_yaml:
                pv = pavi_task_id
                update_ret[k].append(pv)
            continue

        if is_update_yaml:
            if len(v) < 3:
                raise ValueError('{} should provid at least 3 attrs'.format(k))
        else:
            pk = 'pavi_' + k

        if value_type == "max_value":
            try:
                if v[1] == '>':
                    if is_update_yaml:
                        pv = sorted(pavi.get_scalar(
                            pavi_task_id, k, 10, order_key='time'), key=lambda x: x.__getitem__('value'))
                        pv = pv[-1]['value']
                    else:
                        pv = sorted(pavi.get_scalar(pavi_task_id, k, 10),
                                    key=lambda x: x.__getitem__('value'))[-1]['value']
                        pavi_ret[pk] = pv
                else:
                    if is_update_yaml:
                        pv = sorted(pavi.get_scalar(pavi_task_id, k, 10, order_key='time'),
                                    key=lambda x: x.__getitem__('value'), reverse=True)
                        pv = pv[-1]['value']
                    else:
                        pv = sorted(pavi.get_scalar(pavi_task_id, k, 10), key=lambda x: x.__getitem__(
                            'value'), reverse=True)[-1]['value']
                        pavi_ret[pk] = pv
            except:
                pv = 'unknow, {} may not exist on pavi'.format(k)
                if not is_update_yaml:
                    pavi_ret['test_life'] = 0
        elif value_type == "last_value":
            try:
                if is_update_yaml:
                    pv = pavi.get_scalar(pavi_task_id, k, 1, order_key='time')
                    pv = pv[-1]['value']
                else:
                    pv = pavi.get_scalar(pavi_task_id, k, 1)[-1]['value']
                    pavi_ret[pk] = pv
            except:
                pv = 'unknow, {} may not exist on pavi'.format(k)
                if not is_update_yaml:
                    pavi_ret['test_life'] = 0
        else:
            logger.error("Please set 'max' or 'last' for the type of value.")

        if is_update_yaml:
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
    
    if is_update_yaml:
        return config, update_ret

    return config, pavi_ret