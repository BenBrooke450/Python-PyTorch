
import time
import torch





#Element-wise multiplication	[1*1, 2*2, 3*3] = [1, 4, 9]	tensor * tensor
#Matrix multiplication	[1*1 + 2*2 + 3*3] = [14]	tensor.matmul(tensor)

tensor = torch.tensor([1, 2, 3])


# Matrix multiplication by hand
# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]

print(value)
#tensor(14)



print(torch.matmul(tensor,tensor))
#tensor(14)



# Can also use the "@" symbol for matrix multiplication, though not recommended
print(tensor @ tensor)
#tensor(14)





#print(torch.matmul(torch.randn(3,2),torch.rand(3,2)))
"""
  File "/Users/benjaminbrooke/PycharmProjects/Python_PyTroch/PyTorch for Deep Learning Bootcamp - Udemy/Section 2 - PyTorch Fundamentals/11.Matrix Multiplication.py", line 39, in <module>
    print(torch.matmul(torch.randn(3,2),torch.rand(3,2)))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
"""





print(torch.matmul(torch.randn(3,2),torch.rand(2,3)))
"""
tensor([[-0.3082,  0.0637, -1.1258],
        [-0.7087, -0.6270, -1.4060],
        [ 0.4155,  0.4280,  0.7319]])
"""


print(torch.matmul(torch.randn(3,2),torch.rand(2,2)))
"""
tensor([[-1.0241, -0.7489],
        [ 0.4965,  0.3318],
        [-1.0471, -0.7014]])
"""

tensor_A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

tensor_B = torch.tensor([[3,2,1],[6,5,4],[7,8,9]])

print(tensor_A.shape, tensor_B.shape)
#torch.Size([3, 3]) torch.Size([3, 3])




tensor_A = torch.tensor([[1,2],[4,5],[7,8]])

tensor_B = torch.tensor([[3,2],[6,5],[7,8]])

print(tensor_A.shape, tensor_B.shape)
#torch.Size([3, 2]) torch.Size([3, 2])





print(tensor_A,tensor_A.T)
"""
tensor([[1, 2],
        [4, 5],
        [7, 8]]) 
tensor([[1, 4, 7],
        [2, 5, 8]])
"""


print(tensor_A.shape,tensor_A.T.shape)
#torch.Size([3, 2]) torch.Size([2, 3])



print(tensor_A.shape,tensor_B.T.shape)
#torch.Size([3, 2]) torch.Size([2, 3])




print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
#Original shapes: tensor_A = torch.Size([3, 2]), tensor_B = torch.Size([3, 2])

print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
#New shapes: tensor_A = torch.Size([3, 2]) (same as above), tensor_B.T = torch.Size([2, 3])

print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
#Multiplying: torch.Size([3, 2]) * torch.Size([2, 3]) <- inner dimensions match

print("Output:\n")


output = torch.matmul(tensor_A, tensor_B.T)

print(output)
"""
Output:

tensor([[  7,  16,  23],
        [ 22,  49,  68],
        [ 37,  82, 113]])
"""


print(f"\nOutput shape: {output.shape}")
#Output shape: torch.Size([3, 3])

















