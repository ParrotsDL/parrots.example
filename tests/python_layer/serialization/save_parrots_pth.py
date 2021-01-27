import torch
import numpy as np
from resnet import resnet50


net = resnet50()
input = torch.randn(1, 3, 224, 224)
output = net(input)
np.save('input.npy', input.numpy())
torch.save(input, 'input.pkl')
torch.save(input, 'input.zip')
torch.save(output, 'output.pkl')
torch.save(output, 'output.zip')
torch.save(net.state_dict(), 'net.pth')
print('Save successfully!')
