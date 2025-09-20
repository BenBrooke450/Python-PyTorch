
# `torch.eq()`

### What it is

* **Element-wise equality comparison** function in PyTorch.
* Compares two tensors (or a tensor and a scalar) **element by element**.
* Returns a tensor of the same shape, filled with boolean values (`True`/`False`).

---

### Function Signature

```python
torch.eq(input, other, *, out=None) → Tensor
```

* **`input`**: a PyTorch tensor.
* **`other`**: another tensor of the same shape, or a scalar.
* **`out`** (optional): output tensor to store the result.

---

### Behavior

1. **Tensor vs Tensor**

   * Compares corresponding elements of two tensors.
   * Both tensors must be the same shape (or broadcastable).

2. **Tensor vs Scalar**

   * Compares every element in the tensor to the scalar.

3. **Output**

   * Returns a **boolean tensor** of the same shape as `input` (or broadcasted shape).

---

### Examples

```python
import torch

# Example 1: Tensor vs Tensor
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 0, 3])
print(torch.eq(a, b))
# tensor([ True, False,  True])

# Example 2: Tensor vs Scalar
c = torch.tensor([4, 5, 6])
print(torch.eq(c, 5))
# tensor([False,  True, False])

# Example 3: Broadcasting
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([1, 4])
print(torch.eq(x, y))
# tensor([[ True, False],
#         [False,  True]])
```

---

### Relationship to Other Comparison Ops

* `torch.eq(a, b)`  ≈  `a == b`
* `torch.ne(a, b)`  ≈  `a != b`
* `torch.gt(a, b)`  ≈  `a > b`
* `torch.ge(a, b)`  ≈  `a >= b`
* `torch.lt(a, b)`  ≈  `a < b`
* `torch.le(a, b)`  ≈  `a <= b`

So, `torch.eq()` is just the explicit functional form of `==` for tensors.

---

### Typical Use Cases

* Checking prediction correctness (e.g., comparing predicted class vs ground truth).
* Masking tensors (create boolean masks to filter or index elements).
* Debugging: verify if two tensors are identical.
* Building metrics: accuracy = `(torch.eq(y_pred, y_true).sum() / len(y_true))`.

---

### Common Pitfalls

1. **Shape mismatch**

   * Tensors must be the same shape (or broadcastable). Otherwise, you’ll get an error.
2. **Comparing floats**

   * Direct equality comparison between floats can be unreliable due to precision issues.
   * Use `torch.allclose()` or `(torch.abs(a − b) < tolerance)` instead.

---

**In summary:**
`torch.eq()` is PyTorch’s element-wise equality operator. It’s equivalent to `==` but useful when you want to be explicit. Best for comparisons in classification tasks, boolean masks, and debugging.

