


import torch


tensor = torch.tensor([3.0,6.0,9.0],dtype = None, #float32 by default
                      device = None, #cpu by default
                      requires_grad=False #
                      )

tensor_float_16 = torch.tensor([3.0,6.0,9.0],dtype = torch.float16)



#or



print(tensor_float_16*tensor)
#tensor([ 9., 36., 81.])


"""
shape - what shape is the tensor? (some operations require specific shape rules)
dtype - what datatype are the elements within the tensor stored in?
device - what device is the tensor stored on? (usually GPU or CPU)
"""


some_tensor =torch.rand(3,4)

print(some_tensor)
"""
tensor([[0.1967, 0.5300, 0.0602, 0.0048],
        [0.0755, 0.6981, 0.2992, 0.6368],
        [0.4227, 0.2494, 0.4047, 0.2247]])
"""

print(f"Datatype of tensor:{some_tensor.dtype}")
#Datatype of tensor:torch.float32

print(f"Datatype of tensor:{some_tensor.size()}") # OR print(f"Datatype of tensor:{some_tensor.shape}")
#Datatype of tensor:torch.Size([3, 4])

print(f"Datatype of tensor:{some_tensor.device}")
#Datatype of tensor:cpu














