import os
import torch
from .utils import build_helper
from .version import build_pytorch_version, build_pytorch_version_number, \
                     build_parrots_version, build_parrots_version_number, \
                     build_torch_path

torch_path = os.path.dirname(torch.__file__)
pytorch_version, parrots_version = build_helper.get_env_version()
pytorch_version_number, parrots_version_number = build_helper.get_env_version_number()


def check_current_env():
    err_msg = ""
    # if build_pytorch_version_number != 0:
    #     if pytorch_version_number == 0:
    #         err_msg += 'Error: build pape in PyTorch, but now use pape in Parrots\n'
    #     elif build_pytorch_version_number != pytorch_version_number:
    #         err_msg += 'Error: build and use pape in different PyTorch version\n'
    #     else:
    #         return True, err_msg
    # elif build_parrots_version_number != 0:
    #     if parrots_version_number == 0:
    #         err_msg += 'Error: build pape in Parrots, but now use pape in Pytorch\n'
    #     elif build_parrots_version_number != parrots_version_number:
    #         err_msg += 'Error: build and use pape in different Parrots version\n'
    #     else:
    #         return True, err_msg

    # err_msg += 'PyTorch version build {} vs current {}\n'.format(
    #     build_pytorch_version, pytorch_version) + \
    #     'Parrots version build {} vs current {}\n'.format(
    #     build_parrots_version, parrots_version)
    # err_msg += 'build torch path: {}\n'.format(build_torch_path)
    # err_msg += 'current torch path: {}\n'.format(torch_path)
    # return False, err_msg
    return True, err_msg
