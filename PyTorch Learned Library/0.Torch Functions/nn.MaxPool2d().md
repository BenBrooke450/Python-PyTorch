

# `torch.nn.MaxPool2d` in PyTorch

`torch.nn.MaxPool2d` is a 2D max pooling layer commonly used in Convolutional Neural Networks (CNNs). It reduces the spatial dimensions (height and width) of input feature maps while keeping the most prominent features.

---

## **Purpose**

1. **Dimensionality Reduction**: Reduces height and width of feature maps to make the model computationally efficient.
2. **Feature Extraction**: Keeps only the most significant values (max values) in each pooling window, capturing the most important features.
3. **Translation Invariance**: Small shifts in the input do not change the pooled output significantly.

---

## **Function Signature**

```python
torch.nn.MaxPool2d(
    kernel_size, 
    stride=None, 
    padding=0, 
    dilation=1, 
    return_indices=False, 
    ceil_mode=False
)
```

**Parameters:**

| Parameter        | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `kernel_size`    | Size of the window to take a max over (int or tuple)                      |
| `stride`         | Stride of the window. Default = `kernel_size`                             |
| `padding`        | Implicit zero padding added to both sides                                 |
| `dilation`       | Spacing between kernel elements                                           |
| `return_indices` | If True, returns max indices along with output (useful for `MaxUnpool2d`) |
| `ceil_mode`      | If True, uses `ceil` instead of `floor` to compute output shape           |

---

## **Basic Example**

```python
import torch
import torch.nn as nn

# Create a MaxPool2d layer
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Sample input tensor (batch_size=1, channels=1, height=4, width=4)
input_tensor = torch.tensor([[[[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9,10,11,12],
                               [13,14,15,16]]]], dtype=torch.float32)

# Apply max pooling
output = max_pool(input_tensor)
print(output)
```

**Output:**

```
tensor([[[[ 6.,  8.],
          [14., 16.]]]])
```

**Explanation:**

* The `2x2` window slides across the input tensor.
* At each 2x2 window, the **maximum value** is selected.
* Output shape is reduced from `(1,1,4,4)` → `(1,1,2,2)`.

---

## **Using `stride` and `padding`**

```python
# Max pooling with kernel_size=2, stride=1, padding=0
max_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

output = max_pool(input_tensor)
print(output)
```

**Output:**

```
tensor([[[[ 6.,  7.,  8.],
          [10., 11., 12.],
          [14., 15., 16.]]]])
```

* Here, stride=1 → overlapping windows.
* Padding=0 → no padding added.

---

## **Return Indices for Unpooling**

```python
# Max pooling with indices
max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

output, indices = max_pool(input_tensor)
print("Output:\n", output)
print("Indices:\n", indices)
```

* `indices` can be used with `nn.MaxUnpool2d` to reconstruct the original size.

---

## **Key Points**

1. `MaxPool2d` reduces spatial dimensions but preserves **important features**.
2. `kernel_size` determines the size of the pooling window.
3. `stride` controls how the window moves across the input.
4. `return_indices=True` is used for **unpooling** in decoder networks (like autoencoders).
5. Useful in CNNs to **compress feature maps** before fully connected layers.

---

## **Summary Table**

| Feature            | MaxPool2d                      |
| ------------------ | ------------------------------ |
| Purpose            | Downsample feature maps        |
| Keeps              | Maximum value in each window   |
| Reduces            | Height × Width of feature maps |
| Stride default     | Equal to kernel size           |
| Can return indices | Yes, for unpooling             |
| Common use         | CNNs, Autoencoders             |


