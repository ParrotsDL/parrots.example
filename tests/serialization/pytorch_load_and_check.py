import torch
import numpy as np
from resnet import resnet50


net = resnet50()
net.load_state_dict(torch.load('net.pth'))
input = torch.load('input.pkl')
np_input = np.load('input.npy')
assert np.array_equal(input.numpy(), np_input)
output = net(input)
res_output = torch.load('output.pkl')
assert torch.allclose(output, res_output, rtol=5e-05, atol=5e-05)
print('Load and check successfully!')
