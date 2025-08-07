import torch

#Zeros and ones

zero = torch.zeros(size = (3,4))
print(zero)
"""
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
"""



print(torch.rand(3,4)*zero)
"""
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
"""






#Create a tensor of all ones

ones = torch.ones(3,4)
print(ones)
"""
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
"""




print(ones.dtype)
#torch.float32



print(zero.dtype)
#torch.float32









