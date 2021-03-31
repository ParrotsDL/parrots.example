import torch
import numpy as np


npy_pytorch = np.load('./pytorch_data/input.npy')
pkl_pytorch = torch.load('./pytorch_data/input.pkl')
assert np.array_equal(npy_pytorch, pkl_pytorch.numpy())

zip_pytorch = torch.load('./pytorch_data/input.zip')
assert torch.allclose(zip_pytorch, pkl_pytorch, rtol=5e-05, atol=5e-05)

zip2_pytorch = torch.load('./pytorch_data/input2.zip')
assert torch.allclose(zip2_pytorch, torch.tensor([0.]))

print('Load and check successfully!')
