import torch

# Example 1: Reshape a 1D tensor into a 2D tensor
tensor1 = torch.arange(6)
reshaped_tensor1 = tensor1.reshape(2, 3)
print("Reshaped Tensor:\n", reshaped_tensor1)
"""
Reshaped Tensor:
 tensor([[0, 1, 2],
        [3, 4, 5]])
"""


# Example 2: Use -1 to infer one dimension
tensor2 = torch.arange(8)
reshaped_tensor2 = tensor2.reshape(2, -1, 2)
print("Reshaped Tensor with inferred dimension:\n", reshaped_tensor2)
"""
Reshaped Tensor with inferred dimension:
 tensor([[[0, 1],
         [2, 3]],

        [[4, 5],
         [6, 7]]])
"""


# Example 3: Reshape a 2D tensor into a 3D tensor
tensor3 = torch.arange(12).reshape(3, 4)
reshaped_tensor3 = tensor3.reshape(2, 3, 2)
print("Reshaped 3D Tensor:\n", reshaped_tensor3)
"""
Reshaped 3D Tensor:
 tensor([[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]]])
"""


# Example 4: Flatten a 2D tensor into a 1D tensor
tensor4 = torch.tensor([[1, 2, 3], [4, 5, 6]])
reshaped_tensor4 = tensor4.reshape(-1)
print("Flattened Tensor:\n", reshaped_tensor4)
"""
Flattened Tensor:
 tensor([1, 2, 3, 4, 5, 6])

"""