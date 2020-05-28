import os
import psutil
import time
import argparse

import torch
from torch.utils.data import Dataset
try:
    from torch.utils.data import PoolDataLoader
except ImportError:
    from pool_dataloader import PoolDataLoader


parser = argparse.ArgumentParser(description='System Test for DataLoader')
parser.add_argument('-t', '--type', default='during_first_epoch', type=str,
                    help='which kind of test, for example "after_first_epoch"')
parser.add_argument('--model', default=1, type=int,
                    help='which model to be used')
parser.add_argument('-r', '--restart', action='store_true',
                    help='mark for restart test')
group = parser.add_mutually_exclusive_group()
group.add_argument('--multi', action='store_true',
                    help='for multi-models test')
group.add_argument('--single', action='store_true',
                    help='for single model test')

all_type = (["during_first_epoch", "dfe", "after_first_epoch", "afe"]
            +["after_several_epoch", "ase", "after_all_epoch", "aae"]
            +["after_shutdown", "as", "shutdown_and_restart", "sar"])


args = parser.parse_args()
assert args.type in all_type, ("First argument must be one of {}, got {}"
                               .format(str(all_type), args.type))
total_model = 2
test_type = args.type
if len(test_type) <= 3:
    if test_type == "dfe":
        test_type = "during_first_epoch"
    elif test_type == "afe":
        test_type = "after_first_epoch"
    elif test_type == "ase":
        test_type = "after_several_epoch"
    elif test_type == "aae":
        test_type = "after_all_epoch"
    elif test_type == "as":
        test_type = "after_shutdown"
    else:
        test_type = "shutdown_and_restart"

p = psutil.Process()

info = "Test {}! Test: " + test_type
if args.multi:
    info += " for multi-models: No.{} / total: {}".format(args.model, total_model)
else:
    info += " for single model"

class MySet(Dataset):

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        out = torch.empty((2, 2)).fill_(index)
        return out


def main():
    dataset = MySet(256)
    train_loader = PoolDataLoader(dataset, batch_size=32, num_workers=4)
    train_iter = iter(train_loader)

    # wait for all subprocesses to start
    time.sleep(1)

    subprocesses_before_training = p.children()
    pids = set([pro.pid for pro in subprocesses_before_training])

    # first epoch
    train(train_iter, pids, True)

    # test for subprocesses after first epoch
    if test_type == "after_first_epoch":
        test(pids)

    # second epoch
    train_iter = iter(train_loader)
    train(train_iter, pids)

    # test for subprocesses after several epoch
    if test_type == "after_several_epoch":
        test(pids)

    # last epoch
    train_iter = iter(train_loader)
    train(train_iter, pids)

    # test for subprocesses after all epoch
    if test_type == "after_all_epoch":
        test(pids)


def train(train_iter, pids, first=False):
    for i, input in enumerate(train_iter):
        input += 1
        if first and i == 2:
            # test for subprocesses during first epoch
            if test_type == "during_first_epoch":
                test(pids)
            # test for after shutdown
            if test_type == "after_shutdown":
                save(pids)
            
            if test_type == "shutdown_and_restart":
                # test for shutdown and restart
                if args.restart:
                    test(pids)
                save(pids)


def test(pids):
    subpro = set([pro.pid for pro in p.children()])
    assert subpro == pids, info.format("Failed")
    print(info.format("Passed"))
    exit()

def save(pids):
    with open("{}.txt".format(args.model), 'w') as f:
        for id in pids:
            f.write("{}\n".format(id))
        f.write(str(p.pid))
    exit()


if __name__ == "__main__":
    main()
