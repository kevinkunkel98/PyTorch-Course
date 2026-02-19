import torch

def outer_add(a, b):
    return a.unsqueeze(1) + b.unsqueeze(0)

a = torch.tensor([1., 2., 3., 4., 5.])
b = torch.tensor([10., 20., 30.])
c = outer_add(a, b)  # Shape (5, 3), c[i,k] = a[i] + b[k]

def broadcast_mul(a, b):
    return a.unsqueeze(3) * b.unsqueeze(1)

a = torch.randn(3, 2, 5)
b = torch.randn(3, 5, 4)
c = broadcast_mul(a, b)  # Shape (3, 2, 5, 4)