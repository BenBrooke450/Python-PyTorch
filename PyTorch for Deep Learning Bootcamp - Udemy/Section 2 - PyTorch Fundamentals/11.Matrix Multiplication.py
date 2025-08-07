
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


