

"""
There are many different tensor datatypes available in PyTorch.

Some are specific for CPU and some are better for GPU.

Getting to know which one can take some time.

Generally if you see torch.cuda anywhere, the tensor is being used for GPU (since Nvidia GPUs use a computing toolkit called CUDA).

The most common type (and generally the default) is torch.float32 or torch.float.

This is referred to as "32-bit floating point".

But there's also 16-bit floating point (torch.float16 or torch.half) and 64-bit floating point (torch.float64 or torch.double).

And to confuse things even more there's also 8-bit, 16-bit, 32-bit and 64-bit integers.

Plus more!

Note: An integer is a flat round number like 7 whereas a float has a decimal 7.0.

The reason for all of these is to do with precision in computing.

Precision is the amount of detail used to describe a number.

The higher the precision value (8, 16, 32), the more detail and hence data used to express a number.

This matters in deep learning and numerical computing because you're making so many operations, the more detail you have to calculate on, the more compute you have to use.

So lower precision datatypes are generally faster to compute on but sacrifice some performance on evaluation metrics like accuracy (faster to compute but less accurate).

"""


import torch

float_32_tensor = torch.Tensor([3.0,6.0,9.0])

print(float_32_tensor)
#tensor([3., 6., 9.])




tensor = torch.tensor([3.0,6.0,9.0],dtype = None, #float32 by default
                      device = None, #cpu by default
                      requires_grad=False #
                      )

tensor_float_26 = torch.tensor([3.0,6.0,9.0],dtype = torch.float16)



#or



float_16_tensor = float_32_tensor.type(torch.float16)





