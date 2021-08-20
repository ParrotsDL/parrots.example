import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler


class MySet(Dataset):

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.empty((), dtype=torch.int64).fill_(index)

def main():
    bs = 10
    data_len = 200

    dataset = MySet(data_len)
    loader = DataLoader(dataset, batch_size=bs,
                        num_workers=2)

    it = iter(loader)
    prefetch_remain = it.prefetch_remain
    for i in range(20):
        it.next()
        if prefetch_remain >0:
            prefetch_remain -= 1
            assert it.prefetch_remain == prefetch_remain, 'test prefetch_remain failed'
        else:
            assert it.prefetch_remain == 0, 'test prefetch_remain failed'


if __name__ == "__main__":
    main()
    print("Test for dataloader prefetch_remain passed!")
