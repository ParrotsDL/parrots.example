import torch
import parrots
import parrots.compute.operator as op
 
def block(x):
    x = op.mulc(x, 3)
    x = op.sin(x)
    op.subc_(x, 2)
    x = op.astype(x, torch.int32)
    op.mulc_(x, 2)
    x = op.astype(x, torch.float32)
    return op.cos(x)
 
def main():
    x = torch.ones((2, 3)).cuda()
    for _ in range(100):
        x = block(x)
    x.ndarray()
 
    parrots.runtime.profile()
    for i in range(10000):
        x = block(x)
    x.ndarray()
 

if __name__ == "__main__":
    main()
