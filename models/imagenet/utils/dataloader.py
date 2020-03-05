import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pape.data as pdata
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # h, w = img.shape[:2]
        # print('size: {}'.format(img._size))
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def build_augmentation(cfg):
    compose_list = []

    if cfg.random_resize_crop:
        compose_list.append(transforms.RandomResizedCrop(cfg.random_resize_crop))
    if cfg.resize:
        compose_list.append(transforms.Resize(cfg.resize))
    if cfg.random_crop:
        if cfg.padding:
            compose_list.append(transforms.RandomCrop(cfg.random_crop, padding=cfg.padding))
        else:
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

    if cfg.cutout:
        compose_list.append(Cutout(n_holes=cfg.cutout.n_holes, length=cfg.cutout.length))

    return transforms.Compose(compose_list)


def build_dataloader(cfg, world_size):
    train_aug = build_augmentation(cfg.train)
    test_aug = build_augmentation(cfg.test)

    if cfg.type and cfg.type == 'cifar10':
        train_dataset = datasets.CIFAR10(root=cfg.train.image_dir,
                                         train=True, download=False, transform=train_aug)
        test_dataset = datasets.CIFAR10(root=cfg.test.image_dir,
                                        train=False, download=False, transform=test_aug)
    else:
        train_dataset = pdata.McDataset(cfg.train.image_dir, cfg.train.meta_file, train_aug)
        test_dataset = pdata.McDataset(cfg.test.image_dir, cfg.test.meta_file, test_aug)

    train_sampler = pdata.DistributedSampler(train_dataset, batch_size=cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)

    test_sampler = pdata.DistributedSampler(test_dataset, round_up=False, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=(test_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=test_sampler)
    return train_loader, train_sampler, test_loader, test_sampler
