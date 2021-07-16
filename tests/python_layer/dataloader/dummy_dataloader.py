import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DummyDataLoader, PoolDataLoader


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
    train_loader_pool = PoolDataLoader(dataset, batch_size=32, num_workers=4)
    train_loader_dummy = DummyDataLoader(dataset, batch_size=32, num_workers=4)

    if not isinstance(train_loader_pool,
                      (DataLoader, torch.utils.data.DataLoader)):
        raise TypeError(f'dataloader must be a pytorch DataLoader, '
                        f'but got {type(train_loader_pool)}')
    else:
        print("Test for pooldataloader passed!")

    if not isinstance(train_loader_dummy,
                      (DataLoader, torch.utils.data.DataLoader)):
        raise TypeError(f'dataloader must be a pytorch DataLoader, '
                        f'but got {type(train_loader_dummy)}')
    else:
        print("Test for dummydataloader passed!")


    print('All tests passed')


if __name__ == "__main__":
    main()
