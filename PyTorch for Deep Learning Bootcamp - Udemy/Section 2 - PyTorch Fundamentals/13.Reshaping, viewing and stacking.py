

"""
Method	One-line description
torch.reshape(input, shape)	Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
Tensor.view(shape)	Returns a view of the original tensor in a different shape but shares the same data as the original tensor.
torch.stack(tensors, dim=0)	Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.
torch.squeeze(input)	Squeezes input to remove all the dimenions with value 1.
torch.unsqueeze(input, dim)	Returns input with a dimension value of 1 added at dim.
torch.permute(input, dims)	Returns a view of the original input with its dimensions permuted (rearranged) to dims.

"""

import torch


x = torch.arange(1,10)
print(x,x.shape)
#tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]) torch.Size([9])


x_reshape = x.reshape(3,3)
print(x_reshape)
"""
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
"""



x_reshape_2 = x.reshape(9,1)
print(x_reshape_2)
"""
tensor([[1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9]])
"""

x_view = x.view(3,3)
print(x_view)
"""
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
"""





print(torch.stack([x,x],dim=0))
"""
tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]])
"""




print(torch.stack([x_view,x_reshape],dim=0))
"""
tensor([[[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],

        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]])
"""


