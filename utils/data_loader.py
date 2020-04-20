import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from utils.memcached_dataset import McDataset
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
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

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


class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, batch_size=None, world_size=None, rank=None, round_up=True):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.batch_size = batch_size
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
        indices = self.gen_new_list()
        return iter(indices)

    def gen_new_list(self):
        # each process shuffle all list with same seed, and pick one piece according to rank
        shuffle_idx = self.dataset.get_shuffle_idx(self.epoch)
        if self.round_up:
            shuffle_idx += shuffle_idx[:(self.total_size - len(shuffle_idx))]
        offset = self.num_samples * self.rank
        shuffle_idx = shuffle_idx[offset:offset + self.num_samples]
        if self.round_up or (not self.round_up and self.rank < self.world_size - 1):
            assert len(shuffle_idx) == self.num_samples

        print('gen_new_list', self.total_size)

        return shuffle_idx

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_loader(cfg, batch_size, workers, senseagent_config, training=True, dataset_type="memcached", socket_path="unix_socket"):
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
    if (dataset_type == "senseagent"):
        data_set = AgentDataset(
            senseagent_config.userkey,
            senseagent_config.namespace,
            senseagent_config.user,
            senseagent_config.ip,
            senseagent_config.port,
            senseagent_config.distcache,
            senseagent_config.blockshuffleread,
            socket_path,
            cfg.image_dir,
            cfg.meta_file,
            cfg.dataset_name,
            cfg.superblock_source,
            cfg.meta_source,
            transforms.Compose(compose_list),
            cfg.reader)
        round_up = True if training else False
        if senseagent_config.blockshuffleread == True:
            data_sampler = DistributedGivenIterationSampler(data_set, round_up=round_up)
        else:
            data_sampler = DistributedSampler(data_set, round_up=round_up)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=(data_sampler is None),
            num_workers=workers,
            pin_memory=True,
            sampler=data_sampler,
            collate_fn=data_set.collate_fn,
            worker_init_fn=data_set.worker_init_fn)
    else:
        data_set = McDataset(cfg.image_dir, cfg.meta_file,
                             transforms.Compose(compose_list), cfg.reader)
        round_up = True if training else False
        data_sampler = DistributedSampler(data_set, round_up=round_up)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=(data_sampler is None),
            num_workers=workers,
            pin_memory=True,
            sampler=data_sampler)
    return data_loader, data_sampler


def build_dataloader(cfg, dataset_type="memcached", total_epoch=1, socket_path1="unix_socket", socket_path2="unix_socket"):
    train_loader, train_sampler = build_loader(
        cfg.train, cfg.batch_size, cfg.workers, cfg.senseagent_config, training=True, dataset_type=dataset_type, socket_path=socket_path1)
    test_loader, test_sampler = build_loader(
        cfg.test, cfg.batch_size, cfg.workers, cfg.senseagent_config, training=False, dataset_type=dataset_type, socket_path=socket_path2)
    return train_loader, train_sampler, test_loader, test_sampler
