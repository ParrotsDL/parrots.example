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


# def cv2_loader(img_buf):
#     img_array = np.frombuffer(img_buf, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img)

#     return img


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
        # if self.reader == 'opencv':
        #     img = cv2_loader(value_buf)
        # elif self.reader == 'pillow':
        assert self.reader == 'opencv' or self.reader == 'pillow', 'reader should be opencv or pillow.'
        img = pil_loader(value_buf)
        

        # # raw-reading
        # with open(filename, 'rb') as value_str:
        #     with Image.open(value_str) as img:
        #         img = img.convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
