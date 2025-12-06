
import torch

torch_ran_a = torch.rand(3,4)
torch_ran_b = torch.rand(3,4)


print(torch_ran_a)
"""
tensor([[0.9022, 0.3633, 0.0426, 0.7856],
        [0.0650, 0.5512, 0.0603, 0.4543],
        [0.4805, 0.4855, 0.3033, 0.7775]])
"""

print(torch_ran_b)
"""
tensor([[0.3564, 0.1667, 0.3206, 0.6097],
        [0.0579, 0.1976, 0.7848, 0.6864],
        [0.4427, 0.9482, 0.5215, 0.1748]])
"""






torch.manual_seed(42)

torch_ran_a = torch.rand(3,4)

torch_ran_b = torch.rand(3,4)

print(torch_ran_a == torch_ran_b)
"""
tensor([[False, False, False, False],
        [False, False, False, False],
        [False, False, False, False]])
"""









torch.manual_seed(42)
torch_ran_a = torch.rand(3,4)


torch.manual_seed(42)
torch_ran_b = torch.rand(3,4)

print(torch_ran_a == torch_ran_b)
"""
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])
"""






