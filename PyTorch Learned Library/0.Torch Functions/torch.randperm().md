

`torch.randperm(n)`
→ **Generates a random permutation of integers from 0 to n−1**, without repeating any number.

So if `n = 5`, it returns something like:

```
tensor([3, 0, 4, 1, 2])
```

This tensor is basically a **shuffled list of indices**.

---

## ** Function Signature**

```python
torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
```

### **Parameters Explained**

| Parameter       | Type                         | Description                                                           |
| --------------- | ---------------------------- | --------------------------------------------------------------------- |
| `n`             | int                          | The upper limit (0 to n−1) of integers to permute.                    |
| `generator`     | `torch.Generator` (optional) | Controls the random number generator (used for reproducibility).      |
| `out`           | Tensor (optional)            | Output tensor to store result. Rarely used.                           |
| `dtype`         | torch.dtype                  | Default: `torch.int64` (integer type). Usually don’t change this.     |
| `device`        | string or torch.device       | CPU (default) or GPU device to create the tensor on.                  |
| `requires_grad` | bool                         | Default `False`. Usually left as-is unless you’re tracking gradients. |

---

## **Return Value**

A **1D tensor** containing a random permutation of integers `[0, 1, 2, ..., n−1]`.

Each number appears exactly **once** — there are no duplicates.

---

## ** Example Usage**

### Example 1 — Basic use

```python
import torch

perm = torch.randperm(10)
print(perm)
```

Output:

```
tensor([3, 0, 9, 5, 1, 8, 4, 6, 7, 2])
```

This gives a random ordering of numbers from 0 → 9.

---

### Example 2 — Shuffling a dataset

```python
X = torch.tensor([[1,1],[1,2],[2,2],[2,3]], dtype=torch.float32)
y = torch.tensor([1,2,2,3], dtype=torch.float32)

indices = torch.randperm(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]
```

Keeps `X` and `y` **aligned** while shuffling both randomly.
This is one of the **most common uses** of `torch.randperm()` in ML.

---

### Example 3 — Random mini-batch sampling

```python
batch_size = 2
perm = torch.randperm(6)

for i in range(0, len(perm), batch_size):
    indices = perm[i:i+batch_size]
    print(indices)
```

Output (random each time):

```
tensor([3, 0])
tensor([2, 1])
tensor([4, 5])
```

Perfect for implementing **custom mini-batch gradient descent**.

---

### Example 4 — Dataset

If you’re working on CUDA:

```python
X = iris.data
y = iris.target

indices = torch.randperm(len(X))

X = X[indices]
y = y[indices]

X_train = torch.tensor(X,dtype=torch.float32)[:130]
y_train = torch.tensor(y,dtype=torch.long)[:130]

X_test = torch.tensor(X,dtype=torch.float32)[130:]
y_test = torch.tensor(y,dtype=torch.long)[130:]
```

This creates the permutation **directly on the GPU**, which avoids CPU–GPU transfers.

---

### Example 5 — Reproducibility

To ensure the same random order every run:

```python
torch.manual_seed(42)
perm = torch.randperm(5)
print(perm)
```

Every time you run this with the same seed, you get the **same permutation**.

---

## **Common Use Cases**

| Use Case                       | Example                                                   |
| ------------------------------ | --------------------------------------------------------- |
| **Shuffle dataset**            | `indices = torch.randperm(len(X))`                        |
| **Create random mini-batches** | Slice `perm` in chunks of batch size                      |
| **Randomized sampling**        | Pick subset of elements: `X[torch.randperm(len(X))[:10]]` |
| **Cross-validation splits**    | Create random folds by slicing shuffled indices           |
| **GPU training**               | Use `device='cuda'` to generate permutations on GPU       |

---

## **Notes & Tips**

1. **No repetition** → Unlike `torch.randint`, it never repeats numbers.
2. **Uniform randomness** → Each possible permutation is equally likely.
3. **Not gradient-tracked** → It’s just random indices, no need for autograd.
4. **Fast** → Implemented efficiently in C++ backend.
5. **Best for dataset shuffling** → It’s the standard PyTorch approach.

---

## **Summary**

| Property        | Description                                                 |
| --------------- | ----------------------------------------------------------- |
| Function        | `torch.randperm(n)`                                         |
| Returns         | 1D tensor with random permutation of integers from 0 to n−1 |
| Output Type     | `torch.int64`                                               |
| Duplicates      | None                                                        |
| Common Use      | Data shuffling, mini-batching                               |
| Reproducibility | Controlled with `torch.manual_seed()`                       |
| Works On        | CPU or GPU                                                  |

---

### 💡 **Quick Example (All-in-One)**

```python
import torch

torch.manual_seed(123)

# Example dataset
X = torch.arange(12).reshape(6, 2)
y = torch.tensor([0, 1, 0, 1, 0, 1])

# Shuffle using randperm
indices = torch.randperm(len(X))
X = X[indices]
y = y[indices]

print("Shuffled X:\n", X)
print("Shuffled y:\n", y)
```

