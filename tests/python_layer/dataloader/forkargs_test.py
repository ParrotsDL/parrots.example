import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


torch.multiprocessing.set_start_method('fork', force=True)

class MySet(Dataset):

    def __init__(self, size):
        self.size = size
        self.content = b'1'*size

    def __len__(self):
        return 12

    def __getitem__(self, index):
        out = torch.empty((2, 2)).fill_(index)
        return out


def test(dataset_size):
    dataset = MySet(dataset_size)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
    for _, _ in enumerate(dataloader):
        pass

    print('Test for fork arguements pickle/zip/transfer (size={}) passed!'.
          format(dataset_size))

def main():
    test(1024)
    time.sleep(1)
    test(219030)
    time.sleep(1)
    test(20820068)
    time.sleep(1)
    test(1024*1024*1024*3)
    time.sleep(1)

    print('All tests passed')


if __name__ == '__main__':
    main()