import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
# from utils.memcached_dataset import McDataset
from utils.senseagent_dataset import AgentDataset
import torchvision.transforms as transforms
import utils.dist_util as dist
import math


class DistributedSampler(Sampler):
    def __init__(self, dataset, world_size=None, rank=None, round_up=True):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.round_up = round_up
        self.epoch = 0

        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.world_size))
        if self.round_up:
            self.total_size = self.num_samples * self.world_size
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        if self.round_up or (not self.round_up
                             and self.rank < self.world_size - 1):
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_loader(cfg, batch_size, workers, training=True):
    compose_list = []
    if training:
        if cfg.random_resize_crop:
            compose_list.append(
                transforms.RandomResizedCrop(cfg.random_resize_crop))
        else:
            compose_list.append(transforms.Resize(cfg.get('resize', 256)))
            compose_list.append(
                transforms.RandomCrop(cfg.get('random_crop', 224)))
    else:
        compose_list.append(transforms.Resize(cfg.get('resize', 256)))
        compose_list.append(transforms.CenterCrop(cfg.get('center_crop', 224)))
    if cfg.mirror:
        compose_list.append(transforms.RandomHorizontalFlip())
    if cfg.colorjitter:
        compose_list.append(transforms.ColorJitter(*cfg.colorjitter))

    compose_list.append(transforms.ToTensor())
    data_normalize = transforms.Normalize(
        mean=cfg.get('mean', [0.485, 0.456, 0.406]),
        std=cfg.get('std', [0.229, 0.224, 0.225]))
    compose_list.append(data_normalize)
    # data_set = McDataset(cfg.image_dir, cfg.meta_file,
    #                      transforms.Compose(compose_list), cfg.reader)
    data_set = AgentDataset(
        "AQBSsq1cPCqjABAAHcm6x74uBgcEX54FXZKMaA==",
        "ysgns",
        "ysgnsfuse",
        "ysg",
        "10.5.9.171",
        8090,
        cfg.image_dir,
        cfg.meta_file,
        transforms.Compose(compose_list),
        cfg.reader)

    round_up = True if training else False
    data_sampler = DistributedSampler(data_set, round_up=round_up)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=(data_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=data_sampler,
        collate_fn=data_set.collate_fn)
    return data_loader, data_sampler


def build_dataloader(cfg):
    train_loader, train_sampler = build_loader(
        cfg.train, cfg.batch_size, cfg.workers, training=True)
    test_loader, test_sampler = build_loader(
        cfg.test, cfg.batch_size, cfg.workers, training=False)
    return train_loader, train_sampler, test_loader, test_sampler
