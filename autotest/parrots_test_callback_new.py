import os
import os.path as osp
import copy
import sys
import yaml
import pavi
import time
import warnings


from autoparrots.utils.fileio import dump


def after_callback_wrapper(config):
    env = os.environ.copy()
    if env.get('PAVI_TASK_ID') is not None:
        pavi_task_id = env['PAVI_TASK_ID']
    else:
        pavi_task_id = env['pavi_task_id']
    pavi_ret = dict()
    for k, v in config.items():
        if k == '__benchmark_pavi_task_id':
            continue
        pk = 'pavi_' + k
        pv = pavi.get_scalar(pavi_task_id, k, 1)[-1]['value']
        pavi_ret[pk] = pv
    config.update(pavi_ret)

    print(yaml.dump(config))


def update_thresh_wrapper(config, framework, model_name):
    time.sleep(5)  # wait 10s for pavi scalar uploaded    
    env = os.environ.copy()
    if env.get('PAVI_TASK_ID') is not None:
        pavi_task_id = env['PAVI_TASK_ID']
    else:
        pavi_task_id = env['pavi_task_id']

    this_dir = osp.dirname(os.path.abspath(__file__))
    configs_dir = osp.join(this_dir, 'configs')
    config_path = osp.join(configs_dir, framework+'.yaml')
    if '__benchmark_pavi_task_id' not in config:
        # raise KeyError('pavi_task_id not provided')
        # make it compatible with old version
        warnings.warn('__benchmark_pavi_task_id not provided', UserWarning)
        config['__benchmark_pavi_task_id'] = []
    
    # TODO(shiguang): check pavi value
    update_ret = copy.deepcopy(config)
    # attr: [thresh, '>/<', '0.5%/1', val1, val2, ...]
    for k, v in config.items():
        if k == '__benchmark_pavi_task_id':
            pv = pavi_task_id
            update_ret[k].append(pv)
        else:
            if len(v) < 3:
                raise ValueError('{} should provid at least 3 attrs'.format(k))
            pv = pavi.get_scalar(pavi_task_id, k, 1, order_key='time')
            pv = pv[-1]['value']
            update_ret[k].append(pv)
            # update thresh
            mean_pv = sum(update_ret[k][3:len(update_ret[k])]) / 1.0*(len(update_ret[k])-3)
            std_pv = float(v[2]) if not v[2].endswith('%') else float(
                v[2][:-1]) * mean_pv * 0.01

            if v[1] == '>':
                update_ret[k][0] = mean_pv - std_pv
            elif v[1] == '<':
                update_ret[k][0] = mean_pv + std_pv
            else:
                raise KeyError('Unsupported operator key')

    config.update(update_ret)

    org_config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    org_config.update({framework+'_'+model_name: config})
    dump(org_config, config_path, file_format='yaml')

    print(yaml.dump(config))


def pre_callback_wapper(config):

    print(yaml.dump(config))


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
    if len(sys.argv) >= 3:
        config = collect_config(sys.argv[1], sys.argv[2])
        if sys.argv[3] == '0':
            pre_callback_wrapper(config)
        elif sys.argv[3] == '1':
            after_callback_wrapper(config)
        elif sys.argv[3] == '2':
            update_thresh_wrapper(config, sys.argv[1], sys.argv[2])
