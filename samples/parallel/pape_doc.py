import os
import re

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])
local_rank = int(os.environ['SLURM_LOCALID'])

node_list = str(os.environ['SLURM_NODELIST'])
node_list = str(os.environ['SLURM_NODELIST'])
node_parts = re.findall('[0-9]+', node_list)
os.environ['MASTER_ADDR'] = f'{node_parts[1]}.{node_parts[2]}.{node_parts[3]}.{node_parts[4]}'
os.environ['MASTER_PORT'] = str(args.port)
os.environ['WORLD_SIZE'] = str(args.world_size)
os.environ['RANK'] = str(args.rank)

dist.init_process_group(backend="nccl")

torch.cuda.set_device(local_rank)

model = nn.Conv2d(3, 8, (3, 3)).cuda()
model = DDP(model, device_ids=[local_rank])
model.train()
optimizer = torch.optim.SGD(model.parameters(), 0.1)

input = torch.rand(2, 3, 5, 5).float().cuda()
output = model(input)
loss = output.sum()

optimizer.zero_grad()
loss.backward()
optimizer.step()
