import torch
import time
import argparse


parser = argparse.ArgumentParser(description='Tensor size setting')
parser.add_argument("--size", type=int, default=500000)
parser.add_argument("--randperm-size", type=int, default=500000)
args = parser.parse_args()


def test(func, *args, **kwargs):
    n_warm_up = kwargs.pop('n_warm_up', 10)
    n_exec = kwargs.pop('n_exec', 1000)
   
    # warm up  
    for i in range(n_warm_up):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    # count time
    start_time = time.time()
    for i in range(n_exec):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    during = (time.time() - start_time) / n_exec
    print('func', func.__name__, 'cost {} us'.format(during * 1e06))


data = args.size
print('random size = {}, randperm size = {}'.format(data, args.randperm_size))

a = torch.rand(data, device='cuda')
test(torch.rand, data, device='cuda')
test(torch.randn, data, device='cuda')
test(torch.randperm, args.randperm_size, device='cuda',
     n_warm_up=10, n_exec=100)
test(torch.randint, 1000, (data,), device='cuda')
test(torch.Tensor.bernoulli_, a)
test(torch.Tensor.cauchy_, a)
test(torch.Tensor.random_, a)
