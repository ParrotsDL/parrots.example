import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .dataset import McDataset


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


def build_dataloader(cfg, world_size):
    train_aug = build_augmentation(cfg.train)
    test_aug = build_augmentation(cfg.test)
    image_dir = os.getenv("IMAGENET_DATASET_PATH")
    image_dir = '/mnt/lustre/share/images/'
    assert image_dir, "Please set IMAGENET_DATASET_PATH for training"
    train_dataset = McDataset(image_dir + "train", image_dir + "meta/train.txt", train_aug)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)

    test_dataset = McDataset(image_dir + "val", image_dir + "meta/val.txt", test_aug)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=(test_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=test_sampler, drop_last=False)
    return train_loader, train_sampler, test_loader, test_sampler
