import sys

import torch

stack_parrots = torch.load('stack_parrots.pkl')
stack_pytorch = torch.load('stack_pytorch.pkl')

assert len(stack_parrots) == len(stack_pytorch)

for idx, val in enumerate(stack_parrots):
    try:
        assert stack_parrots[idx] < stack_pytorch[idx]
    except Exception:
        if stack_parrots[idx] - stack_pytorch[idx] / stack_parrots[idx] < 0.5:
            pass
        else:
            print("Test failed!")
            sys.exit()

print("Test successfully!")
