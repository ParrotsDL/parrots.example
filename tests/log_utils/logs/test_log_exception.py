import torch

import parrots
import os
import multiprocessing as mp

fname = 'log_exception.txt'

if os.path.exists(fname):
    os.remove(fname)

def test_exception(f=False):

    if f:
        parrots.log_utils.log_to_file(fname, False, False)

    parrots.log_utils.log_info('info 0')

    a = torch.ones((2, 2))
    b = torch.ones((3, 3))

    # an exception
    try:
        c = a + b
    except ValueError as e:
        parrots.log_utils.log_info('get an ValueError:\n {}'.format(str(e)))


if __name__ == '__main__':
    p1 = mp.Process(target=test_exception)
    p2 = mp.Process(target=test_exception, args=(True,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
