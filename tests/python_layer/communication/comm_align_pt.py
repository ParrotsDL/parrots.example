import logging
import torch
from torch import device, distributed as dist

logging.basicConfig(level=logging.INFO)

assert torch.cuda.is_available()

world_size = dist.get_world_size()
assert world_size == 4

odds = [rank for rank in range(world_size) if rank % 2 == 1]
assert odds == [1, 3]
even = [rank for rank in range(world_size) if rank % 2 == 0]
assert even == [0, 2]
group_odds = dist.new_group(odds, backend='nccl')
group_even = dist.new_group(even, backend='nccl')

x = torch.tensor([2.0, 2.0], device='cuda')
if dist.get_rank() in [2, 3]:
    y = torch.ones((2, ), device='cuda')
else:
    y = torch.zeros((2, ), device='cuda')

if dist.get_rank() in [0, 1]:
    z = torch.zeros((2, ), device='cuda')
else:
    z = torch.ones((2, ), device='cuda')

dist.all_reduce(x, group=group_odds)
dist.all_reduce(x, group=group_even)
assert x.sum().item() == 8.0
dist.broadcast(y, 3, group=group_odds)
dist.broadcast(z, 1, group=group_odds)
dist.broadcast(y, 2, group=group_even)
dist.broadcast(z, 0, group=group_even)
assert torch.add(y, z).sum().item() == 2.0

x = torch.tensor(1.0 * dist.get_rank()).cuda()
tensor_list_odds = [torch.zeros(()).cuda() for i in range(2)]
tensor_list_even = [torch.zeros(()).cuda() for i in range(2)]
dist.gather(x, tensor_list_odds, 3, group_odds)
dist.gather(x, tensor_list_even, 2, group_even)
if dist.get_rank() == 3:
    assert torch.add(tensor_list_odds[0], tensor_list_odds[1]).sum().item() == 4.0
if dist.get_rank() == 2:
    assert torch.add(tensor_list_even[0], tensor_list_even[1]).sum().item() == 2.0

logging.info("[PASSED] Communication Test")
