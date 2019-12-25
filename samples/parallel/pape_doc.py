import torch
import torch.nn as nn
import pape
from pape.parallel import DistributedModel as DM

pape.distributed.init()

model = nn.Conv2d(3, 8, (3, 3)).cuda()
model = DM(model)
model.train()
optimizer = torch.optim.SGD(model.parameters(), 0.1)

input = torch.rand(2, 3, 5, 5).float().cuda()
output = model(input)
loss = output.sum()

optimizer.zero_grad()
loss.backward()
optimizer.step()
