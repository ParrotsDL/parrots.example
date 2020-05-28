import torch

import parrots
import os
import glob

from parrots import version

fname = 'log_limit.txt'
for f in glob.glob(os.path.join('.', 'log_limit*.txt')):
    os.remove(f)

parrots.log_utils.log_to_file(fname, False, False)
parrots.log_utils.change_log_file_size(1000)

for i in range(100):
    parrots.log_utils.log_info(f'{i}th message is here!')
