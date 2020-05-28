import torch

def main():
    a = torch.ones(2,3)
    a.requires_grad = True
    a.backward()

main()
