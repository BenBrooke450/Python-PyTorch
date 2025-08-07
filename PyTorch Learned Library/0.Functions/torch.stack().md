
### Summary of `torch.stack()`

- **Purpose:** Concatenates a sequence of tensors along a new dimension.

- **Syntax:**
  ```python
  torch.stack(tensors, dim=0, out=None)
  ```

- **Parameters:**
  - `tensors`: A sequence of tensors of the same shape.
  - `dim`: The index of the new dimension along which the tensors are stacked. Default is `0`.
  - `out` (optional): An output tensor to store the result.

- **Returns:** A tensor with the input tensors stacked along the specified dimension.

### Examples of `torch.stack()`

Here are several examples demonstrating how to use `torch.stack()`:

```python
import torch

# Example 1: Stacking 1D tensors along a new dimension
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
stacked_tensor1 = torch.stack([tensor1, tensor2])
print("Stacked 1D Tensors:\n", stacked_tensor1)
"""
Stacked 1D Tensors:
 tensor([[1, 2, 3],
        [4, 5, 6]])
"""


# Example 2: Stacking 2D tensors along a new dimension
tensor3 = torch.tensor([[1, 2], [3, 4]])
tensor4 = torch.tensor([[5, 6], [7, 8]])
stacked_tensor2 = torch.stack([tensor3, tensor4])
print("Stacked 2D Tensors along dim=0:\n", stacked_tensor2)
"""
Stacked 2D Tensors along dim=0:
 tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
"""


# Example 3: Stacking along a different dimension
stacked_tensor3 = torch.stack([tensor3, tensor4], dim=1)
print("Stacked 2D Tensors along dim=1:\n", stacked_tensor3)
"""
Stacked 2D Tensors along dim=1:
 tensor([[[1, 2],
         [5, 6]],

        [[3, 4],
         [7, 8]]])
"""


# Example 4: Stacking tensors with an additional dimension
tensor5 = torch.tensor([1, 2])
tensor6 = torch.tensor([3, 4])
tensor7 = torch.tensor([5, 6])
stacked_tensor4 = torch.stack([tensor5, tensor6, tensor7], dim=1)
print("Stacked Tensors along dim=1:\n", stacked_tensor4)
"""
Stacked Tensors along dim=1:
 tensor([[1, 3, 5],
        [2, 4, 6]])
"""


# Example 5: Stacking tensors with an additional dimension
tensor5 = torch.tensor([1, 2])
tensor6 = torch.tensor([3, 4])
tensor7 = torch.tensor([5, 6])
stacked_tensor4 = torch.stack([tensor5, tensor6, tensor7], dim=0)
print("Stacked Tensors along dim=0:\n", stacked_tensor4)
"""
Stacked Tensors along dim=0:
 tensor([[1, 2],
        [3, 4],
        [5, 6]])
"""


# Example 6: Stacking tensors with different data types
tensor8 = torch.tensor([1.0, 2.0])
tensor9 = torch.tensor([3.0, 4.0])
stacked_tensor5 = torch.stack([tensor8, tensor9])
print("Stacked Tensors with float data type:\n", stacked_tensor5)
"""
Stacked Tensors with float data type:
 tensor([[1., 2.],
        [3., 4.]])
"""
```

### Explanation of the Examples

- **Example 1:** Stacks two 1-dimensional tensors along a new dimension, resulting in a 2-dimensional tensor.

- **Example 2:** Stacks two 2-dimensional tensors along the default dimension (`dim=0`), resulting in a 3-dimensional tensor.

- **Example 3:** Stacks the same two 2-dimensional tensors along a different dimension (`dim=1`), resulting in a different 3-dimensional tensor structure.

- **Example 4:** Stacks three 1-dimensional tensors along `dim=1`, resulting in a 2-dimensional tensor.

- **Example 5:** Stacks two 1-dimensional tensors with a floating-point data type, resulting in a 2-dimensional tensor with the same data type.

`torch.stack()` is a versatile function for combining tensors along new dimensions, which is useful in various scenarios such as preparing batches of data for machine learning models.