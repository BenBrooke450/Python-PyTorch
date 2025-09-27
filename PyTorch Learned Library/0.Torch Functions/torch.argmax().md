
# **`torch.argmax()` Summary**

### **Definition**

```python
torch.argmax(input, dim=None, keepdim=False)
```

Returns the **indices of the maximum values** of a tensor along a given dimension.

---

# **Parameters**

| Parameter | Description                                                                                                                                                                                                                   |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input`   | The input tensor (any shape, any dtype that supports comparison).                                                                                                                                                             |
| `dim`     | The dimension along which to compute the index of max value. <br> - If `None` (default), flattens the tensor and returns the index in the **flattened 1D view**. <br> - If integer, returns max indices along that dimension. |
| `keepdim` | If `True`, the output tensor has the same number of dimensions as input (with size `1` in the reduced dim). If `False`, that dimension is squeezed out.                                                                       |

---

# **Return Value**

* A tensor of type `torch.int64` (indices).
* Shape depends on `dim` and `keepdim`.

---

# **Examples**

### 1. **Flattened (default)**

```python
import torch

x = torch.tensor([[1, 3, 2],
                  [4, 6, 5]])

print(torch.argmax(x))  
# tensor(4)

print(x.flatten()[4])  
# tensor(6)
```

Here it flattens to `[1,3,2,4,6,5]` → max is `6` at index `4`.

---

### 2. **Along a row (dim=1)**

```python
print(torch.argmax(x, dim=1))
# tensor([1, 1])
```

* Row 0 → `[1,3,2]` → max at index `1`
* Row 1 → `[4,6,5]` → max at index `1`

---

### 3. **Along a column (dim=0)**

```python
print(torch.argmax(x, dim=0))
# tensor([1, 1, 1])
```

* Col 0 → `[1,4]` → max at row 1 → index `1`
* Col 1 → `[3,6]` → max at row 1 → index `1`
* Col 2 → `[2,5]` → max at row 1 → index `1`

---

### 4. **Keep dimensions**

```python
print(torch.argmax(x, dim=1, keepdim=True))
# tensor([[1],
#         [1]])
```

Shape preserved: `(2,1)` instead of `(2,)`.

---

### 5. **1D vector**

```python
v = torch.tensor([10, 25, 7, 25])
print(torch.argmax(v))  
# tensor(1)
```

👉 Note: if multiple max values exist (`25` at index 1 and 3), **the first occurrence is returned**.

---

### 6. **GPU usage**

```python
x = torch.tensor([1.0, 9.0, 3.0], device="cuda")
print(torch.argmax(x))  
# tensor(1, device='cuda:0')
```

---

# **Comparison with Related Functions**

| Function         | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| `torch.argmax()` | Returns index of max value                              |
| `torch.argmin()` | Returns index of min value                              |
| `torch.max()`    | Returns the max values **and optionally their indices** |
| `torch.topk()`   | Returns top-k values and indices                        |

Example:

```python
values, indices = torch.max(x, dim=1)
```

---

# **Deep Learning Use Case**

In classification:

```python
logits = torch.tensor([[2.1, 0.9, 3.2],
                       [1.5, 4.2, 0.7]])

preds = torch.argmax(logits, dim=1)
print(preds)  
# tensor([2, 1]) → class labels
```

---

✅ **Summary**

* `torch.argmax()` → indices of maximum values.
* Default flattens the tensor.
* Use `dim` to control direction (rows, cols, etc.).
* First max wins if duplicates exist.
* Critical in classification to convert raw model outputs into predicted labels.

