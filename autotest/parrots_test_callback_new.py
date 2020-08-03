import os
import os.path as osp
import sys
import yaml
import pavi

def after_callback_wapper(config):
    env = os.environ.copy()
    if env.get('PAVI_TASK_ID') is not None:
        pavi_task_id = env['PAVI_TASK_ID']
    else:
        pavi_task_id = env['pavi_task_id']
    pavi_ret = dict()
    for k, v in config.items():
        pk = 'pavi_' + k
        pv = pavi.get_scalar(pavi_task_id, k, 1)[-1]['value']
        pavi_ret[pk] = pv
    config.update(pavi_ret)

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
        if sys.argv[3] == 'pre':
            pre_callback_wapper(config)
        else:
            after_callback_wapper(config)
