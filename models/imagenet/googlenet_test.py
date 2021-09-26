import torch
import models
from torch.utils import quantize

model = models.__dict__["googlenet"](num_classes=1000, aux_logits=True)
model = quantize.convert_to_adaptive_quantize(model, 1000)
