import torch
# import parrots
# import parrots.compute.operator as op
 
def block(x):
    # x = op.mulc(x, 3)
    x = torch.mul(x, 3)
    # x = op.sin(x)
    x = torch.sin(x)
    # op.subc_(x, 2)
    x = torch.sub(x, 2)
    x = x.type(torch.int32)
    x = torch.mul(x, 2)
    # x = op.astype(x, torch.float32)
    x = x.type(torch.float32)
    x = torch.cos(x)
    return x
 
def main():
  with torch.autograd.profiler.profile(use_cuda=True) as prof:
    x = torch.ones((2, 3)).cuda()
    for _ in range(100):
        x = block(x)
    # x.ndarray()
    x = x.cpu().numpy()
    x = torch.tensor(x).cuda()
 
    # parrots.runtime.profile()
    for i in range(10000):
        x = block(x)
    # x.ndarray()
    x = x.cpu().numpy()
  print(prof.key_averages().table(sort_by="self_cpu_time_total"))
 

if __name__ == "__main__":
    main()
