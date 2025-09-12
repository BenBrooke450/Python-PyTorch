

## **1. `torch.min()`**

### **Purpose:**

* Returns the **minimum value** of a tensor.
* Can also return **element-wise minimum** between two tensors.

---

### **Syntax & Uses:**

1. **Global minimum of a tensor:**

```python
import torch

x = torch.tensor([1, 3, 2, 5])
min_val = torch.min(x)
print(min_val)  # tensor(1)
```

2. **Minimum along a dimension (axis):**

```python
x = torch.tensor([[1, 2, 3],
                  [4, 0, 6]])

min_val, min_idx = torch.min(x, dim=0)  # along columns
print(min_val)  # tensor([1, 0, 3])
print(min_idx)  # tensor([0, 1, 0])  → index of min in each column
```

3. **Element-wise minimum between two tensors:**

```python
a = torch.tensor([1, 4, 3])
b = torch.tensor([2, 2, 5])
c = torch.min(a, b)
print(c)  # tensor([1, 2, 3])
```

---

### **Key Points:**

* `torch.min(tensor)` → returns single value (if tensor is 1D or flattened).
* `torch.min(tensor, dim=?)` → returns tuple `(values, indices)`.
* `torch.min(tensor1, tensor2)` → element-wise minimum.

---

<br><br><br><br>

## **2. `torch.max()`**

### **Purpose:**

* Returns the **maximum value** of a tensor.
* Can also return **element-wise maximum** between two tensors.

---

### **Syntax & Uses:**

1. **Global maximum of a tensor:**

```python
x = torch.tensor([1, 3, 2, 5])
max_val = torch.max(x)
print(max_val)  # tensor(5)
```

2. **Maximum along a dimension (axis):**

```python
x = torch.tensor([[1, 2, 3],
                  [4, 0, 6]])

max_val, max_idx = torch.max(x, dim=1)  # along rows
print(max_val)  # tensor([3, 6])
print(max_idx)  # tensor([2, 2])  → index of max in each row
```

3. **Element-wise maximum between two tensors:**

```python
a = torch.tensor([1, 4, 3])
b = torch.tensor([2, 2, 5])
c = torch.max(a, b)
print(c)  # tensor([2, 4, 5])
```

---

### **Key Points:**

* `torch.max(tensor)` → single maximum value.
* `torch.max(tensor, dim=?)` → tuple `(values, indices)` along a dimension.
* `torch.max(tensor1, tensor2)` → element-wise maximum.

---

### **Quick Comparison Table**

| Function    | Single value | Along dim          | Element-wise between tensors |
| ----------- | ------------ | ------------------ | ---------------------------- |
| `torch.min` | ✅            | ✅ (values+indices) | ✅                            |
| `torch.max` | ✅            | ✅ (values+indices) | ✅                            |







