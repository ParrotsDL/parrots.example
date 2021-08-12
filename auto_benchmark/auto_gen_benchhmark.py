# import sys, os
# sys.path.insert(0, os.getcwd())

from fileio import load
from utils import FileTemplate
import os


def build_target_file(config):
    # parse from source yaml
    for key, value in config.items():
        target_templates = value['templates']
        target_value = []
        for i in value['pro'].split(','):
            path_yaml = os.path.abspath(
                os.path.dirname(__file__)) + "/benchmark_{}.yaml".format(i)
            cfg = load(path_yaml, file_format='yaml')
            target_value += cfg

        target_file = FileTemplate(key, target_templates, target_value)

        # build target ap yaml
        # target_path = os.path.abspath(
        #     os.path.dirname(__file__)) + "/BENCHMARK_{}.yaml".format(key)
        # target_file.build_yaml(target_path=target_path)

        # build target sh
        target_path = os.path.abspath(
            os.path.dirname(__file__)) + "/BENCHMARK_{}.sh".format(key)
        target_file.build_shell(target_path=target_path)


if __name__ == "__main__":
    path = os.path.abspath(
        os.path.dirname(__file__)) + "/benchmark_config.yaml"
    cfg = load(path, file_format='yaml')
    build_target_file(cfg)
