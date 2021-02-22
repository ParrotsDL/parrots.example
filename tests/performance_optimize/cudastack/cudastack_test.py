import time

import torch


if torch.__version__ == "parrots":
    import parrots
    parrots.runtime.exec_mode = 'SYNC'

res = []


def test(method):
    if method == 0:
        a = torch.randn(16, 32, 14204, 4).cuda()
        b = torch.randn(16, 32, 14204, 4).cuda()
        c = torch.randn(16, 32, 14204, 4).cuda()
    elif method == 1:
        a = torch.randn(16, 32, 14, 4).cuda()
        b = torch.randn(16, 32, 14, 4).cuda()
        c = torch.randn(16, 32, 14, 4).cuda()
    elif method == 2:
        a = torch.randn(16, 32, 14204, 40).cuda()
        b = torch.randn(16, 32, 14204, 40).cuda()
        c = torch.randn(16, 32, 14204, 40).cuda()
    else:
        a = torch.randn(2, 11, 16, 64).cuda()
        b = torch.randn(2, 11, 16, 64).cuda()
        c = torch.randn(2, 11, 16, 64).cuda()

    # warm up
    for i in range(10):
        torch.stack([a, b, c], dim=-2, out=None)

    # test
    torch.cuda.synchronize()
    for dim in range(0, 5):
        shape = list(a.shape)
        shape.insert(dim, 3)
        out = torch.empty(size=shape, device='cuda', dtype=a.dtype)
        tensors = [a, b, c]
        t = time.time()
        for i in range(100):
            torch.stack(tensors, dim=dim, out=out)
        torch.cuda.synchronize()
        t = time.time() - t
        print('stack at dim {}, shape {}, time cost per-iter: {}'
              .format(dim, tuple(shape), t / 100))
        res.append(t / 100)


for m in range(0, 4):
    test(m)

if torch.__version__ == "parrots":
    savefile = 'stack_parrots.pkl'
else:
    savefile = 'stack_pytorch.pkl'
torch.save(res, savefile)
