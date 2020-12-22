"""
1. 这里只放置callback的主要逻辑，具体各部分功能细节放在callback_util下

2. autoparrots的各个函数以及其他第三方包在callback_util/__init__.py中导入

3. 自己添加的工具包在callback_util/__init__.py中导入, callback_util目录下的文件以callback_开头, \
   具体参考callback_common, callback_monitor, callback_utils.
"""
import os
import os.path as osp
import copy
import sys
import yaml
import pavi
import time

from callback_util import (callback_common, callback_monitor, callback_utils,
                            logger, dump, trace_up)


class CallBack(object):

    def __init__(self, framework, model, run_type):
        self.framework = framework
        self.model = model
        self.run_type = run_type  # all / benchmark / dailytest and so on
        assert self.run_type in callback_common.run_type_table.keys()

        # get config from {framework} and {model}
        self.config = self._collect_config(self.framework, self.model)

        if self.framework in callback_common.value_type_table.keys():
            self.value_type = callback_common.value_type_table[self.framework]
        else:
            # default: read last_value
            self.value_type = callback_common.value_type_table["default"]

    def run_pre_callback_wrapper(self, is_monitor_log=True):
        config = self.config
        run_type = self.run_type
        framework = self.framework
        model = self.model

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
        config['test_life'] = 0
        if run_type == 'autoparrotsbenchmark':
            if config['__benchmark_total_time(h)'] == [10000, '<', '5%']:
                del config['__benchmark_total_time(h)']
        else:
            config['__benchmark_total_time(h)'] = 10000

        # get slurm job id
        slurm_job_id = ''
        status = ''
        start_time = time.time()
        name = os.environ['name']
        while True:
            interval_time = time.time() - start_time
            if interval_time >= callback_common.wait_time_get_slurm_jobid * 60 * 60:
                logger.warn(
                    "Job({}): can't get slurm from squeue".format(name))
                break
            slurm_job_id, _, status = callback_utils.get_slurm_job_id()
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
                if interval_time >= callback_common.wait_time_fork_subprocess * 60 * 60:
                    break
                pid = os.fork()
            if pid == 0:
                callback_monitor.watch_for_kill_time_limited(
                    framework, model, config)

    def run_after_callback_wrapper(self, output_monitor=True):
        config = self.config
        value_type = self.value_type
        run_type = self.run_type

        # get scalar from pavi and update ret dict
        config, pavi_ret = callback_utils.get_scalar_from_pavi(
            config, value_type, run_type)

        config.update(pavi_ret)
        print(yaml.dump(config))

        if output_monitor and config['test_life'] == 1:
            monitor_info = callback_utils.get_monitor_info(config, run_type)
            dump(monitor_info, 'monitor_info.json')
            from insertdata import DataInseter
            data_inster = DataInseter()
            data_inster.insert(**monitor_info)

    def run_get_benchmark_value(self):
        config = self.config
        framework = self.framework
        model = self.model
        value_type = self.value_type
        run_type = self.run_type

        # get scalar from pavi and update ret dict
        config, pavi_ret = callback_utils.get_scalar_from_pavi(
            config, value_type, run_type)

        print(yaml.dump(pavi_ret))
        # TODO(zhouhanyu): get pavi benchmark value
        config_path = self._find_framework_config('benchmark')
        if os.path.exists(config_path):
            org_config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
            org_config.update({framework+'_' + model: pavi_ret})
        else:
            org_config = {framework+'_' + model: pavi_ret}
        dump(org_config, config_path, file_format='yaml', default_flow_style=False)


    def run_update_thresh_wrapper(self):
        config = self.config
        framework = self.framework
        model = self.model
        value_type = self.value_type
        run_type = self.run_type

        time.sleep(5)  # wait 10s for pavi scalar uploaded

        config_path = self._find_framework_config('configs')
        full_config = copy.deepcopy(config)

        # get scalar from pavi and update ret dict
        config, update_ret = callback_utils.get_scalar_from_pavi(
            config, value_type, run_type, is_update_yaml=True)

        full_config[run_type] = update_ret
        config = full_config

        org_config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        org_config.update({framework+'_'+model: config})
        dump(org_config, config_path, file_format='yaml', default_flow_style=False)

        config = config[run_type]
        print(yaml.dump(config))

    def _collect_config(self, framework, model):
        this_dir = osp.dirname(os.path.abspath(__file__))
        configs_dir = osp.join(this_dir, 'configs')
        configs = dict()
        for f in os.listdir(configs_dir):
            config_path = osp.join(configs_dir, f)
            configs.update(
                **yaml.load(open(config_path, 'r'), Loader=yaml.Loader))
        key = framework + '_' + model
        return configs[key]

    def _find_framework_config(self, dirname):
        framework = self.framework
        root_path = trace_up('.search-run')
        root_path = root_path.replace('.search-run', 'autotest')
        configs_dir = osp.join(root_path, dirname)
        config_path = osp.join(configs_dir, framework+'.yaml')
        return config_path


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        callbacker = CallBack(
            framework=sys.argv[1], model=sys.argv[2], run_type=sys.argv[4])
        if sys.argv[3] == '0':
            callbacker.run_pre_callback_wrapper()
        elif sys.argv[3] == '1':
            callbacker.run_after_callback_wrapper()
        elif sys.argv[3] == '2':
            callbacker.run_update_thresh_wrapper()
        elif sys.argv[3] == '3':
            callbacker.run_get_benchmark_value()
