
# Summary of `torch.rand()`

- **Purpose:** Creates a tensor filled with random numbers sampled from a uniform distribution over the interval [0, 1).

- **Syntax:**
  ```python
  torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
  ```

- **Parameters:**
  - `*size`: A sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.
  - `out` (optional): An output tensor to store the result.
  - `dtype` (optional): The desired data type of the tensor. If not specified, it defaults to `torch.float`.
  - `layout` (optional): The desired layout of the tensor. Default is `torch.strided`.
  - `device` (optional): The desired device of the tensor (e.g., `cpu` or `cuda`). Default is `None`, which uses the current device.
  - `requires_grad` (optional): If autograd should record operations on the returned tensor. Default is `False`.

- **Returns:** A tensor filled with random values from a uniform distribution over [0, 1), with the specified shape and data type.

### Example of `torch.rand()`

Here's a code example demonstrating how to use `torch.rand()`:

```python

import torch

# Example 1: Create a 1D tensor with random values
tensor1 = torch.rand(5)
print("1D Random Tensor:", tensor1)
#1D Random Tensor: tensor([0.0583, 0.1376, 0.2400, 0.7173, 0.8114])


# Example 2: Create a 2D tensor with random values
tensor2 = torch.rand(3, 4)
print("2D Random Tensor:\n", tensor2)
"""
2D Random Tensor:
 tensor([[0.7749, 0.8600, 0.6454, 0.6143],
        [0.0490, 0.5282, 0.0360, 0.0821],
        [0.2280, 0.8995, 0.4161, 0.9785]])
"""


# Example 3: Create a 3D tensor with random values and a specified data type
tensor3 = torch.rand((2, 3, 4), dtype=torch.double)
print("3D Random Tensor with double data type:\n", tensor3)
"""
3D Random Tensor with double data type:
 tensor([[[0.0484, 0.4635, 0.4976, 0.8483],
         [0.4742, 0.5421, 0.8775, 0.7795],
         [0.7244, 0.5120, 0.1504, 0.7810]],

        [[0.2116, 0.4974, 0.1400, 0.9935],
         [0.6768, 0.3647, 0.5157, 0.4255],
         [0.8479, 0.7248, 0.0701, 0.4268]]], dtype=torch.float64)
"""

```

### Explanation of the Example

- **Example 1:** Creates a 1-dimensional tensor with random values, with a length of 5.

- **Example 2:** Creates a 2-dimensional tensor with random values, with 3 rows and 4 columns.

- **Example 3:** Creates a 3-dimensional tensor with random values, with a shape of (2, 3, 4) and specifies the data type as `torch.double`.

- **Example 4:** Creates a 2-dimensional tensor with random values on a CUDA device if available. This demonstrates how you can specify the device for the resulting tensor.

`torch.rand()` is a useful function for generating tensors with random values, which is often necessary in machine learning for tasks like weight initialization or data augmentation.






<br><br><br>


# Example 1
```python
import torch


print(torch.rand(7,7))
"""
tensor([[0.1986, 0.7301, 0.9671, 0.4492, 0.8930, 0.5844, 0.0144],
        [0.0259, 0.8436, 0.9880, 0.9896, 0.0458, 0.3509, 0.1182],
        [0.8584, 0.3690, 0.7425, 0.5828, 0.2551, 0.3252, 0.9754],
        [0.3683, 0.2476, 0.2561, 0.0494, 0.2019, 0.6999, 0.1061],
        [0.7403, 0.1754, 0.2495, 0.9502, 0.7742, 0.8340, 0.0554],
        [0.7935, 0.3104, 0.7661, 0.3600, 0.9148, 0.0253, 0.4718],
        [0.6049, 0.8337, 0.2666, 0.5096, 0.3654, 0.3045, 0.1053]])
"""




print(torch.mm(torch.rand(7,7),torch.rand(1,7).T))
"""
tensor([[1.8814],
        [2.2381],
        [1.6810],
        [1.8336],
        [2.6028],
        [1.8462],
        [2.3004]])
"""



torch.manual_seed(0)
print(torch.mm(torch.rand(7,7),torch.rand(1,7).T))
"""
tensor([[1.8542],
        [1.9611],
        [2.2884],
        [3.0481],
        [1.7067],
        [2.5290],
        [1.7989]])
"""



print(torch.mm(torch.rand(7,7),torch.rand(1,7).T).reshape(1,7))
#tensor([[1.2239, 2.0847, 1.9002, 0.9408, 1.5213, 1.3606, 0.8780]])
```

