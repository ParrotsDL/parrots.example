import yaml
from io import StringIO
from fileio import dump
import logging
import os
import copy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


template_benchmark = """
task:
    # Required (change partition, num_gpus or others where necessary)
    template: ""
    # Required
    search_args:
        - type: Nested
          value: [
                    # (un)commit your config here
                ]

    # Optional
    # callback: "cat"
    pre_callback: "python autotest/parrots_test_callback_new.py {framework} {model} 0 benchmark"
    after_callback: "python autotest/parrots_test_callback_new.py {framework} {model} 1 benchmark"

searcher:
    # Support GridSearcher, RandomSearcher, SkoptSearcher
    type: GridSearcher
    # Required
    reward_attribute: acc

scheduler:
    type: FIFO
    parallel_size: 8
    stop_criterion:
        max_tries: 250
    wait_to_run_time_thresh: inf # inf
    wait_to_get_pavi_task_thresh: 0.5

# Optional
pavi:
    compare:
        name: parrots.test.newbenchmark
"""


class FileTemplate(object):
    def __init__(self, method, templates, target_value):
        self.templates = templates
        self.target_value = target_value
        self.method = method

    def build_yaml(self, target_path=None):
        template_file = yaml.load(StringIO(template_benchmark),
                                  Loader=yaml.Loader)

        self.target_yaml = template_file
        self.target_yaml['task']['search_args'][0]['value'] = self.target_value
        self.target_yaml['task']['template'] = self.templates

        # dump
        dump(self.target_yaml, file=target_path, file_format='yaml')

    def build_shell(self, target_path=None, merge=True):
        # build whole commmand
        f_srun_args = []


        c_config = os.getenv("MULTI_CARD", 0)

        if c_config:
            new_config = []
            c_config = c_config.split(",")            
            for i in c_config:
                temp_target = copy.deepcopy(self.target_value)
                for var in temp_target:
                    if var['gpus']!=int(i):
                        var['gpus']=int(i)
                        new_config.append(var)
            self.target_value +=new_config

        target_command = [
            self.templates.format(**var) + '\n'for var in self.target_value
        ]

        srun_args = os.getenv("SRUN_ARGS", '')
        if srun_args:
            if "SH-IDC1" in srun_args:
                sa = srun_args.split(' ')
                for idx, item in enumerate(sa):
                    if "SH-IDC1" in item and ',' in item.split('[')[-1]:
                        item=item.split(']')[0]
                        card = [var['gpus'] for var in self.target_value]
                        for c in card:
                            if c <= 8:
                                sa[idx]="".join(item.split('[')[0]) + item.split('[')[-1].split(',')[0]
                            elif c<=16 and c> 8:
                                sa[idx]="".join(item.split('[')[0]) + '['+','.join(item.split('[')[-1].split('-')[0:2])+']'
                            elif c<=24 and c>16:
                                sa[idx]="".join(item.split('[')[0]) + '['+','.join(item.split('[')[-1].split('-')[0:3])+']'
                            elif c<=32 and c>24:
                                sa[idx]="".join(item.split('[')[0]) + '['+','.join(item.split('[')[-1].split('-')[0:4])+']'
                            else:
                                logging.info(f"{c} card is incompatible for from srun_args.")
                            srun_args = " ".join(sa)
                            f_srun_args.append("SRUN_ARGS=" + "\"" + srun_args + "\"" +" ")
                        target_command = [f_srun_args[idx] + target_command[idx] for idx in range(len(target_command))]
                    elif "SH-IDC1" in item and ',' not in item.split('[')[-1]:
                        f_srun_args="SRUN_ARGS=" + "\"" + srun_args + "\"" +" "
                        target_command = [f_srun_args + target_command[idx] for idx in range(len(target_command))]
                    else:
                        pass


        if merge:
            target_command.insert(0, f'##{self.method}\n')
            merge_path = "/".join(target_path.split("/")[:-1]) + "/merged_0.sh"
            with open(merge_path, "a+") as f:
                f.writelines(target_command)
            logging.info(f'{self.method} sh merged to {merge_path} done')
        else:
            with open(target_path, "w") as f:
                f.writelines(target_command)
            logging.info(f'build {target_path} done')
