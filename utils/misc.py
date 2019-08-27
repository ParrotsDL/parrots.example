import torch
import utils.dist_util as dist_util


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        batch_size = target.size(0)
        full_batch_size = torch.Tensor([batch_size]).to(output)
        world_size = dist_util.get_world_size()
        if world_size > 1:
            dist_util.all_reduce(full_batch_size)
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).to(output).sum(0, keepdim=True)
            if world_size > 1:
                dist_util.all_reduce(correct_k)
            res.append(correct_k.mul_(100.0 / full_batch_size))
        return res


def logger(msg, rank=0):
    world_size = dist_util.get_world_size()
    global_rank = dist_util.get_rank()
    assert rank < world_size
    if rank >= 0:
        if global_rank == rank:
            print(msg)
    elif rank == -1:
        print(msg)
