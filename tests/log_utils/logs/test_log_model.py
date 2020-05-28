import torch

import parrots
import os
import glob
import socket

from parrots import version

torch.cuda.set_device(torch.distributed.get_rank())
fname = 'log_model.txt'

for f in glob.glob(os.path.join('.', 'log_model*.txt')):
    os.remove(f)

def test_model(f=False):
    if f:
        parrots.log_utils.log_to_file(fname, True, False)

    parrots.log_utils.set_debug_log(True)
    parrots.log_utils.log_info('model name: Test Model')
    parrots.log_utils.log_debug(f'version: {version.git_hash}')
    parrots.log_utils.log_warn(f'compute version: {version.compute.git_hash}')
    hostname = socket.gethostname()
    parrots.log_utils.log_debug(f'hostname: {hostname}')
    dev_id = torch.cuda.current_device()
    parrots.log_utils.log_info(f'gpu id: {dev_id}')


if __name__ == '__main__':
    test_model()
    test_model(True)
