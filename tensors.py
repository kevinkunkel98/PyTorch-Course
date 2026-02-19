import torch

a= torch.full((2,3),3.)
b= torch.full((5,1,3),3.)
c= a+b

print(a)
print(b)