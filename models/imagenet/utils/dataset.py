import copy
import io
import os

from PIL import Image
use_mc = False
try:
   import mc
   use_mc = True
except:
    print("import mc failed")
from torch.utils.data import Dataset

class McDataset(Dataset):
    r"""
    Dataset using memcached to read data.

    Arguments
        * root (string): Root directory of the Dataset.
        * meta_file (string): The meta file of the Dataset. Each line has a image path
          and a label. Eg: ``nm091234/image_56.jpg 18``.
        * transform (callable, optional): A function that transforms the given PIL image
          and returns a transformed image.
    """
    def __init__(self, root, meta_file, transform=None):
        self.root = root
        self.transform = transform
        with open(meta_file) as f:
            meta_list = f.readlines()
        self.num = len(meta_list)
        self.metas = []
        self.ddt_flag = 1
        self.ddt_img = 0
        for line in meta_list:
            path, cls = line.strip().split()
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

    def __getitem__(self, index):
        if os.environ.get('DUMMYDATASET') == '1':
            if self.ddt_flag:
                filename = self.root + '/' + self.metas[index][0]
                cls = self.metas[index][1]

                # memcached
                self._init_memcached()
                value = mc.pyvector()
                self.mclient.Get(filename, value)
                value_buf = mc.ConvertBuffer(value)
                buff = io.BytesIO(value_buf)
                with Image.open(buff) as img:
                    img = img.convert('RGB')
                self.ddt_img = copy.deepcopy(img)
                self.ddt_flag = 0
                print("*********no dummydataset*********")
            else:
                img = copy.deepcopy(self.ddt_img)
                cls = self.metas[index][1]
                print("#######dummydataset#######")
        else:
            filename = self.root + '/' + self.metas[index][0]
            cls = self.metas[index][1]

            # memcached
            if use_mc:
                self._init_memcached()
                value = mc.pyvector()
                self.mclient.Get(filename, value)
                value_buf = mc.ConvertBuffer(value)
                buff = io.BytesIO(value_buf)
                with Image.open(buff) as img:
                    img = img.convert('RGB')
            else:
                with Image.open(filename) as img:
                    img = img.convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
