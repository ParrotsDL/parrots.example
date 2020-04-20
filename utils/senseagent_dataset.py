from SenseAgentClient import SenseAgentClientNG as sa
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataloader import default_collate
import io
import cv2
from PIL import Image
from utils.misc import logger
import sys

g_batch_size = 1024


def pil_loader2(b_data):
    buff = io.BytesIO(b_data)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


def cv2_loader2(img_buf):
    img_array = np.frombuffer(img_buf, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


class AgentDataset(Dataset):
    def __init__(self, userKey, nameSpace, user, agentIp, agentPort, enableDistCache, blockShuffleRead, socket_path, root_dir, meta_file, dataSet, superblock_file, superblock_meta, transform=None, reader='pillow'):
        self.root_dir = root_dir
        self.transform = transform
        self.reader = reader
        self.agentclient = None
        self.sacli_for_shuffle = None
        self.userKey = userKey
        self.nameSpace = nameSpace
        self.dataSet = dataSet
        self.user = user
        self.agentIp = agentIp
        self.agentPort = agentPort
        self.enableDistCache = enableDistCache
        self.blockShuffleRead = blockShuffleRead
        self.socket_path = socket_path
        print("fuck socket_path ", socket_path)
        self.superblock_file = superblock_file
        self.superblock_meta = superblock_meta
        self.in_list = []
        self.image_idx = {}
        self.shuffle_idx = []
        self.metas = []
        self.metas_shuffle = []
        self.initialized = False

        with open(meta_file) as f:
            lines = f.readlines()
        self.num = len(lines)
        counter = 0
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
            short_image = path.rsplit("/", 1)[-1]
            self.in_list.append(short_image)
            self.image_idx[short_image] = counter
            counter = counter + 1

    def __init_senseagent(self):
        if not self.initialized:
            self.agentclient = sa.SenseAgent(self.userKey, self.nameSpace, self.dataSet,
                                             self.user, self.agentIp, self.agentPort, self.blockShuffleRead, False)
            if self.blockShuffleRead:
                # self.agentclient.loadMetainfos(self.superblock_meta)
                self.agentclient.setBlockShuffleParameter(
                    self.in_list, self.superblock_meta, g_batch_size, "",  self.socket_path)
            if self.enableDistCache:
                self.agentclient.loadMetainfos(self.superblock_meta)
                my_rank = self.agentclient.startDistCache(0.1)
                print("my rank is", my_rank)
            self.initialized = True

    def __len__(self):
        return self.num

    def _set_shuffle_idx(self, epoch):
        self.shuffle_idx = []
        sacli_for_shuffle = sa.SenseAgent(self.userKey, self.nameSpace, self.dataSet,
                                          self.user, self.agentIp, self.agentPort, self.blockShuffleRead, True)
        sacli_for_shuffle.loadMetainfos(self.superblock_meta)
        sacli_for_shuffle.setBlockShuffleParameter(self.in_list, self.superblock_meta, g_batch_size, "testshuffle", "")
        out_list = sacli_for_shuffle.generateBlockShuffleRandomFileList(g_batch_size, epoch)
        for out in out_list:
            self.shuffle_idx.append(self.image_idx[out])

        del sacli_for_shuffle

    def get_shuffle_idx(self, epoch):
        self._set_shuffle_idx(epoch)
        return self.shuffle_idx

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        return filename, cls

    def collate_fn(self, data):
        self.__init_senseagent()
        file_names = [elem[0] for elem in data]
        #print("collate_fn len ", len(file_names))
        # sys.stdout.flush()
        img_pools = self.agentclient.batchReadFile(file_names)
        if self.reader == 'opencv':
            trans_pools = [self.transform(cv2_loader2(img)) for img in img_pools]
        elif self.reader == 'pillow':
            trans_pools = [self.transform(pil_loader2(img)) for img in img_pools]
        else:
            assert self.reader == 'opencv' or self.reader == 'pillow', 'reader should be opencv or pillow.'
        cls_pools = [elem[1] for elem in data]
        img_cls_pool = list(zip(trans_pools, cls_pools))
        # print(file_names)
        return default_collate(img_cls_pool)

    def worker_init_fn(self, worker_id):
        print("worker_init_fn ", worker_id)
        self.__init_senseagent()
