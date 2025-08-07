


"""
These operations are often a wonderful dance between:

Addition
Substraction
Multiplication (element-wise)
Division
Matrix multiplication

"""
import torch

tensor = torch.tensor([1,2,3,4,5,6,7])

print(tensor + 10)
#tensor([11, 12, 13, 14, 15, 16, 17])



print(tensor*10)
#tensor([10, 20, 30, 40, 50, 60, 70])



print(tensor - 10)
#tensor([-9, -8, -7, -6, -5, -4, -3])




print(torch.mul(tensor,10))
#tensor([10, 20, 30, 40, 50, 60, 70])








