import torch
import sys
import numpy as np

model_path = sys.argv[1]
print("Check model: ", model_path)
m = torch.load(model_path)

if 'state_dict' in m:
    m = m['state_dict']

nan_c = 0
ok_c = 0
for k, v in m.items():
    if np.any(np.isnan(v.cpu().numpy())):
        print("{} has nan".format(k))
        nan_c += 1
    else:
        print("{} OK".format(k))
        ok_c += 1
print("Summary, {} has nan, {} OK".format(nan_c, ok_c))
