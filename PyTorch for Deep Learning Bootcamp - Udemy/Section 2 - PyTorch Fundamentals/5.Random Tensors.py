



"""
We've established tensors represent some form of data.

And machine learning models such as neural networks manipulate and seek patterns within tensors.

But when building machine learning models with PyTorch, it's rare you'll create tensors by hand (like what we've been doing).

Instead, a machine learning model often starts out with large random tensors of numbers and adjusts these random numbers as it works through data to better represent it.

In essence:

Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers...

"""

# https://docs.pytorch.org/docs/stable/generated/torch.rand.html


import torch

# Random tensors


random_tensor = torch.rand(3,4)
print(random_tensor)
"""
tensor([[0.2542, 0.5173, 0.1424, 0.8828],
        [0.8715, 0.3627, 0.0225, 0.4811],
        [0.3401, 0.8493, 0.8978, 0.6130]])
"""

print(random_tensor.ndim)
#2





random_tensor = torch.rand(3,4,3)
print(random_tensor)
"""
tensor([[[0.1408, 0.2880, 0.9297],
         [0.3850, 0.2910, 0.1722],
         [0.1133, 0.5926, 0.8019],
         [0.2772, 0.8279, 0.8025]],

        [[0.3410, 0.1098, 0.1226],
         [0.5344, 0.7049, 0.7930],
         [0.5866, 0.2328, 0.6937],
         [0.9543, 0.5235, 0.6359]],

        [[0.7181, 0.9151, 0.4454],
         [0.2199, 0.7891, 0.2300],
         [0.4661, 0.9987, 0.9515],
         [0.4981, 0.8741, 0.5789]]])
"""

print(random_tensor.ndim)
#3








random_image_size_tensor = torch.rand(size = (224,224,3))

print(random_image_size_tensor.shape)
#torch.Size([224, 224, 3])

print(random_image_size_tensor.ndim)
#3
























