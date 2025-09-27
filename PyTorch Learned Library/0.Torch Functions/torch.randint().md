


# **`torch.randint()` Summary**

### **Definition**

```python
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
```

Returns a tensor filled with **random integers** uniformly sampled from the range `[low, high)`.

---

## 1. Parameters

| Parameter                | Description                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `low` *(int, default=0)* | Inclusive lower bound of the random integers.                                                   |
| `high` *(int)*           | Exclusive upper bound of the random integers. Required.                                         |
| `size` *(tuple of ints)* | Shape of the output tensor.                                                                     |
| `generator` *(optional)* | A `torch.Generator` for controlling RNG state (e.g., reproducibility).                          |
| `out` *(optional)*       | Output tensor to write results into (in-place).                                                 |
| `dtype` *(torch.dtype)*  | Desired data type of output tensor. Default: `torch.int64`.                                     |
| `layout`                 | Layout of the returned tensor (default: `torch.strided`). Rarely changed.                       |
| `device` *(optional)*    | Device where tensor is allocated (`'cpu'`, `'cuda'`, etc.).                                     |
| `requires_grad` *(bool)* | If `True`, autograd will track operations on the returned tensor. Usually `False` for integers. |

---

## 2. Behavior

* **Range:** `[low, high)` (low inclusive, high exclusive).
* **Uniform distribution:** Every integer in the range has equal probability.
* **No guarantee of uniqueness** (values can repeat).
* **Default dtype:** `torch.int64`.

---

## 3. Examples

### Basic usage

```python
import torch

x = torch.randint(0, 10, (3, 4))
print(x)
# tensor([[3, 1, 7, 9],
#         [0, 6, 2, 8],
#         [5, 4, 1, 3]])
```

* Integers from `0` to `9`.
* Shape `(3,4)`.

---

### One argument for `high`

If you omit `low`, it defaults to `0`:

```python
x = torch.randint(5, (2, 2))
print(x)  # values in [0, 5)
```

---

### Specify dtype

```python
x = torch.randint(0, 100, (5,), dtype=torch.int32)
print(x.dtype)  # torch.int32
```

---

### With generator (reproducibility)

```python
g = torch.Generator().manual_seed(42)
x = torch.randint(0, 10, (2, 3), generator=g)
print(x)
```

Running this again with the same seed produces the same tensor.

---

### Out parameter

```python
out_tensor = torch.empty(2, 3, dtype=torch.int64)
torch.randint(0, 10, (2, 3), out=out_tensor)
print(out_tensor)
```

---

## 4. Related Functions

| Function               | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `torch.rand()`         | Random floats in `[0,1)`.                                      |
| `torch.randn()`        | Random floats from standard normal distribution.               |
| `torch.randint_like()` | Same as `torch.randint()` but shape taken from another tensor. |
| `torch.randperm()`     | Random permutation of integers `[0, n)`, no repeats.           |

---

## 5. Common Use Cases

* Initializing **random integer data** (labels, IDs).
* Creating **synthetic test data**.
* Sampling random **indices** for shuffling or batching.
* Quick creation of **binary/random masks** (`torch.randint(0, 2, size)` → 0/1).

---

## Summary

* `torch.randint(low, high, size)` → integers in `[low, high)`.
* Default `low=0`.
* Returns uniform random integers.
* Shape controlled by `size`.
* Dtype default = `int64`, but can be changed.
* Supports `device`, reproducibility via `generator`, and writing into an existing tensor.

