import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# torch.tenosr() =  https://docs.pytorch.org/docs/stable/tensors.html


#Scalar
scalar = torch.tensor(7)
print(scalar)
#tensor(7)



print(scalar.ndim)
#0


scalar_item = scalar.item()
print(scalar_item)
#7









#Vector
vector = torch.tensor([7,7])
print(vector)
#tensor([7, 7])



print(vector.shape)
#torch.Size([2])








#Matrix
matrix = torch.tensor([[7,8],[9,10]])
print(matrix)
"""
tensor([[ 7,  8],
        [ 9, 10]])
"""


print(matrix.ndim)
#2

print(matrix.shape)
#torch.Size([2, 2])


print(matrix[1])
#tensor([ 9, 10])



#Tensor
tensor = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])

print(tensor)
"""
tensor([[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]])
"""



print(tensor.ndim)
#3


print(tensor.shape)
#torch.Size([1, 3, 3])


print(tensor[0])
"""
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
"""






































