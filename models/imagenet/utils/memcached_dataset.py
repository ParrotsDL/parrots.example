import mc
from torch.utils.data import Dataset
import io
from PIL import Image
import cv2
import numpy as np


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')

    return img


class McDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, reader='pillow'):
        self.root_dir = root_dir
        self.transform = transform
        self.reader = reader
        with open(meta_file) as f:
            lines = f.readlines()
        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
        self.initialized = False

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]

        # memcached
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_buf = mc.ConvertBuffer(value)
        assert self.reader == 'pillow', 'reader should be pillow.'
        img = pil_loader(value_buf)
        

        # # raw-reading
        # with open(filename, 'rb') as value_str:
        #     with Image.open(value_str) as img:
        #         img = img.convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
