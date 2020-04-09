from SenseAgentClient import SenseAgentClientNG as sa
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataloader import default_collate
import io
import cv2
from PIL import Image

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
    def __init__(self, userKey, nameSpace, user, agentIp, agentPort, enableDistCache, blockShuffleRead, root_dir, meta_file, dataSet, superblock_file, superblock_meta, transform=None, reader='pillow'):
        self.root_dir = root_dir
        self.transform = transform
        self.reader = reader
        self.agentclient = None
        self.userKey = userKey
        self.nameSpace = nameSpace
        self.dataSet = dataSet
        self.user = user
        self.agentIp = agentIp
        self.agentPort = agentPort
        self.enableDistCache = enableDistCache
        self.blockShuffleRead = blockShuffleRead
        self.superblock_file = superblock_file
        self.superblock_meta = superblock_meta
        self.in_list = []
        self.initialized = False
        with open(meta_file) as f:
            lines = f.readlines()
        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
            short_image = path.rsplit("/", 1)[-1]
            self.in_list.append(short_image)
        
    def __init_senseagent(self):
        if not self.initialized:
            self.agentclient = sa.SenseAgent(self.userKey, self.nameSpace, self.dataSet, self.user, self.agentIp, self.agentPort, self.blockShuffleRead)
            if self.blockShuffleRead:
                self.agentclient.loadMetainfos()
                self.agentclient.setBlockShuffleParameter(self.in_list, 16)
            if self.enableDistCache:
                self.agentclient.loadMetainfos()
                my_rank = self.agentclient.startDistCache(0.5)
                print("my rank is", my_rank)
            self.initialized = True
	
    def __len__(self):
        return self.num

          
    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        return filename, cls

    def collate_fn(self, data):
        self.__init_senseagent()
        file_names = [elem[0] for elem in data]
        img_pools = self.agentclient.batchReadFile(file_names)
        if self.reader == 'opencv':
            trans_pools = [self.transform(cv2_loader2(img)) for img in img_pools]
        elif self.reader == 'pillow':
            trans_pools = [self.transform(pil_loader2(img)) for img in img_pools]
        else:
            assert self.reader == 'opencv' or self.reader == 'pillow', 'reader should be opencv or pillow.'
        cls_pools = [elem[1] for elem in data]
        img_cls_pool = list(zip(trans_pools, cls_pools))
        return default_collate(img_cls_pool)
