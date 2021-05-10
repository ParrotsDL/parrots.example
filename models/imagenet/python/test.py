import torch

a = torch.randn(2, 3, 224, 224)
a = a.contiguous(torch.channels_last)
a = a.cuda()

# b = torch.flatten(a, 1)
c = a.view(a.size(0), -1) 
# print(b.shape, b.stride(), b[0][:5])
# print(c.shape, c.stride(), c[0][:5])
# b = a.permute(0, 2, 3, 1).contiguous()
# # b = a.contiguous(torch.channels_last)

# print("a", a.shape, a.stride(), a.is_contiguous(), a.flatten()[:3])
# print("b", b.shape, b.stride(), b.is_contiguous(), b.flatten()[:3])

# b = b.contiguous()
# print("b", b.shape, b.stride(), b.is_contiguous(), b.flatten()[:3])
