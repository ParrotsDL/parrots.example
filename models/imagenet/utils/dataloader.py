import torch
import torchvision.transforms as transforms
import pape.data as pdata


def build_augmentation(cfg):
    compose_list = []
    if cfg.random_resize_crop:
        compose_list.append(transforms.RandomResizedCrop(cfg.random_resize_crop))
    else:
        compose_list.append(transforms.Resize(cfg.get('resize', 256)))
        compose_list.append(transforms.RandomCrop(cfg.get('random_crop', 224)))

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

    train_dataset = pdata.McDataset(cfg.train.image_dir, cfg.train.meta_file, train_aug)
    train_sampler = pdata.DistributedSampler(train_dataset, batch_size=cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)

    test_dataset = pdata.McDataset(cfg.test.image_dir, cfg.test.meta_file, test_aug)

    test_sampler = pdata.DistributedSampler(test_dataset, round_up=False, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=(test_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=test_sampler)
    return train_loader, train_sampler, test_loader, test_sampler
