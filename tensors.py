import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

shape = (2,3)

ones = torch.ones(shape, device=device)
zeros = torch.zeros(shape, device=device)
random = torch.rand(shape, device=device)

print(ones)
print(ones.device)

