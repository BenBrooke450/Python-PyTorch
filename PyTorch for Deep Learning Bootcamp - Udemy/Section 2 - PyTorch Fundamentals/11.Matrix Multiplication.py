
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
value






