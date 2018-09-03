import torch
a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
b = torch.Tensor(a).float()
for i in a:
    print i