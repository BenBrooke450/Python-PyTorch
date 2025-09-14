

# 🔹 `state_dict()` in PyTorch

## 🔸 What it is

* A **Python dictionary object** that maps each **layer name** (string) → its **parameters (tensor values)**.
* Contains **all learnable weights and biases** of the model (and sometimes buffers like running stats in batch norm).
* Used for **saving** and **loading** models in a portable way.

---

## 🔸 Where It’s Used

1. **Model Parameters**

   ```python
   model.state_dict()   # dictionary of all parameters
   ```

2. **Optimizer State**

   ```python
   optimizer.state_dict()  # dictionary with momentum, learning rate, etc.
   ```

---

## 🔸 Example: Model State Dict

```python
import torch.nn as nn
import torch

# Simple model
model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)

# Show state_dict
print(model.state_dict())
```

Output (example):

```
OrderedDict([
  ('0.weight', tensor([[...], [...]])),
  ('0.bias', tensor([..])),
  ('2.weight', tensor([[...]])),
  ('2.bias', tensor([..]))
])
```

Here:

* `"0.weight"` → weights of first Linear layer.
* `"0.bias"` → bias of first Linear layer.
* `"2.weight"` / `"2.bias"` → from the last Linear layer.

---

## 🔸 Example: Optimizer State Dict

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(optimizer.state_dict())
```

Output (simplified):

```
{
 'state': { ... }, 
 'param_groups': [{'lr': 0.01, 'momentum': 0.9, ...}]
}
```

---

## 🔸 Saving & Loading with `state_dict()`

### Save model + optimizer:

```python
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}, "checkpoint.pth")
```

### Load later:

```python
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
```

---

## 🔸 Why `state_dict()` is Better than Saving the Full Model

* ✅ Portable (just weights, not class definitions).
* ✅ Works even if you restructure code slightly.
* ✅ Common best practice in PyTorch.

---

✅ **Summary:**
`state_dict()` returns a dictionary containing all parameters (weights & biases) and persistent buffers of a model (or optimizer). It’s the **preferred way to save and load models** in PyTorch because it’s lightweight, portable, and flexible.
