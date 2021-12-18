import torch

use_camb = False
if torch.__version__ == "parrots":
    from parrots.base import use_camb

int_dtype = torch.int if use_camb else torch.long
