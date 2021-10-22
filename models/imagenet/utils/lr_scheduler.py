import math


def adjust_learning_rate_cos(optimizer, epoch, iteration, num_iter, args):
    lr = optimizer.param_groups[0]['lr']
    warmup_epoch = 3
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.max_epoch * num_iter

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter
    else:
        lr = args.lr * (1 + math.cos(math.pi * (current_iter - warmup_iter) /
                                     (max_iter - warmup_iter))) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
