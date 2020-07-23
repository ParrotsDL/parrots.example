import os
import os.path as osp
import sys
import yaml


def callback_wapper(config):
    # '__autotest_get_scalar_from_pavi': if true autoparrots will
    # get scalar from pavi
    config.update(dict(__autotest_get_scalar_from_pavi=True))
    # Required: encoding to yaml format and output to stdout,
    #           so that it can be auto-parsed to summary.
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
        callback_wapper(config)
