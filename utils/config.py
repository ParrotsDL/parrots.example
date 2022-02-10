import torch

use_camb = False
use_hip = False
if torch.__version__ == "parrots":
    from parrots.base import use_camb, use_hip

int_dtype = torch.int if use_camb else torch.long
