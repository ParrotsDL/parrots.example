import torch
import torchvision.transforms as transforms
import pape.data as pdata
def build_augmentation(cfg):
    compose_list = []
    if cfg.random_resize_crop:
        compose_list.append(transforms.RandomResizedCrop(cfg.random_resize_crop))
    if cfg.resize:
        compose_list.append(transforms.Resize(cfg.resize))
    if cfg.random_crop:
        compose_list.append(transforms.RandomCrop(cfg.random_crop))
    if cfg.center_crop:
        compose_list.append(transforms.CenterCrop(cfg.center_crop))

    if cfg.mirror:
        compose_list.append(transforms.RandomHorizontalFlip())
    if cfg.colorjitter:
        compose_list.append(transforms.ColorJitter(*cfg.colorjitter))

    compose_list.append(transforms.ToTensor())

    data_normalize = transforms.Normalize(mean=cfg.get('mean', [0.485, 0.456, 0.406]),
                                          std=cfg.get('std', [0.229, 0.224, 0.225]))
    compose_list.append(data_normalize)

    return transforms.Compose(compose_list)

# def get_item(mem_dataset, dataset, sub_indices):
#     for i in sub_indices:
#         mem_dataset[i] = dataset[i]

# class MemDataset(Dataset):
#     def __init__(self, mem_dataset, transform=None,  indices=None):
#         self.dataset = mem_dataset
#         self.transform = transform
#         self.indices = indices

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         try:
#             img, cls = self.dataset[index]
#         except Exception as e:
#             raise(e)
#             # print(index," ", self.dataset[index])
#         # img = copy.deepcopy(img)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img,cls


def load_dataset_to_mem(dataset, indices, transform=None, workers=8):
    import threading
    mem_dataset = list(range(len(dataset)))
    # thread_list = []
    # dataset_len = len(indices)
    for i in range(1, workers+1):
        t = threading.Thread(target=get_item, args=(mem_dataset, dataset, indices[int(
            dataset_len/workers*(i-1)):int(dataset_len/workers*(i))]))
        t.start()
        thread_list.append(t)
    for i in thread_list:
        i.join()
    mem_dataset = MemDataset(mem_dataset, transform, indices)
    return mem_dataset


def build_dataloader(cfg, world_size):
    train_aug = build_augmentation(cfg.train)
    test_aug = build_augmentation(cfg.test)

    train_dataset = pdata.McDataset(cfg.train.image_dir, cfg.train.meta_file, train_aug)
    train_sampler = pdata.DistributedSampler(train_dataset, batch_size=cfg.batch_size, manual_seed=99)
    train_indices = list(train_sampler.__iter__())
    train_dataset._parallel_warmup_mem(train_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=False, sampler=train_sampler)

    test_dataset = pdata.McDataset(cfg.test.image_dir, cfg.test.meta_file, test_aug)
    test_sampler = pdata.DistributedSampler(test_dataset, round_up=False, manual_seed=99)
    test_indices = list(test_sampler.__iter__())
    test_dataset._parallel_warmup_mem(test_indices)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=(test_sampler is None),
        num_workers=0, pin_memory=True, sampler=test_sampler)
    return train_loader, train_sampler, test_loader, test_sampler
