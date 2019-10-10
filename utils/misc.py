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


def build_syncbn(model, group, **kwargs):
    for name, mod in model.model.named_modules():
        if len(name) == 0:
            continue
        parent_module = model
        for mod_name in name.split('.')[0:-1]:
            parent_module = getattr(parent_module, mod_name)
        last_name = name.split('.')[-1]
        last_module = getattr(parent_module, last_name)
        if isinstance(last_module, torch.nn.BatchNorm2d):
             print('replace module {} with syncbn.'.format(name))
             syncbn = dist_util.SyncBatchNorm2d(last_module.num_features,
                                              eps=last_module.eps,
                                              momentum=last_module.momentum,
                                              affine=last_module.affine,
                                              group=group,
                                              **kwargs)
             if last_module.affine:
                 syncbn.weight.data = last_module.weight.data.clone().detach()
                 syncbn.bias.data = last_module.bias.data.clone().detach()
             syncbn.running_mean.data = last_module.running_mean.clone().detach()
             syncbn.running_var.data = last_module.running_var.clone().detach()
             parent_module.add_module(last_name, syncbn)
    return model


def logger(msg, rank=0):
    world_size = dist_util.get_world_size()
    global_rank = dist_util.get_rank()
    assert rank < world_size
    if rank >= 0:
        if global_rank == rank:
            print(msg)
    elif rank == -1:
        print(msg)
