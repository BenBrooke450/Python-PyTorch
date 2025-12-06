


import torch

m = torch.rand(3,4)

print(m)
"""
tensor([[0.6601, 0.8205, 0.8578, 0.1085],
        [0.9548, 0.6483, 0.7816, 0.7129],
        [0.5968, 0.6626, 0.0357, 0.6321]])
"""


#OR


m_r = torch.arange(0,9).reshape(3,3)
print(m_r)
"""
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
"""


print(m_r[0])
#tensor([0, 1, 2])

print(m_r[:,0])
#tensor([0, 3, 6])

print(m_r[:,1:])
"""
tensor([[1, 2],
        [4, 5],
        [7, 8]])
"""




