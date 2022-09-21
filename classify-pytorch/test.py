import torch

print('Has CUDA:', torch.cuda.is_available())
print('Random array:', torch.rand(5, 3))
