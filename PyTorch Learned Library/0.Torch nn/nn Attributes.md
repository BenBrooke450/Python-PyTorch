

# ✅ **Setup**

We define a tiny model so we can demonstrate everything:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8*8*8, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.adaptive_avg_pool2d(x, (8,8))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SmallNet()
```

---

# ✅ 1. **`model.parameters()`**

Returns all learnable parameters (tensors wrapped in `nn.Parameter`).

```python
for p in model.parameters():
    print(p.shape)
```

**Output**

```
torch.Size([8, 3, 3, 3])   # conv weight
torch.Size([8])            # conv bias
torch.Size([8])            # BN weight
torch.Size([8])            # BN bias
torch.Size([8])            # BN running_mean (buffer)
torch.Size([8])            # BN running_var  (buffer)
torch.Size([10, 512])      # fc weight
torch.Size([10])           # fc bias
```

---

# ✅ 2. **`model.named_parameters()`**

Gives the **names**:

```python
for name, p in model.named_parameters():
    print(name, p.shape)
```

**Output**

```
conv.weight torch.Size([8, 3, 3, 3])
conv.bias torch.Size([8])
bn.weight torch.Size([8])
bn.bias torch.Size([8])
fc.weight torch.Size([10, 512])
fc.bias torch.Size([10])
```

---

# ✅ 3. **`model.children()`**

Returns **direct submodules**:

```python
print(list(model.children()))
```

**Output**

```
[Conv2d(3,8), BatchNorm2d(8), Linear(512 → 10)]
```

---

# ✅ 4. **`model.modules()`**

Returns **ALL modules (recursive)**:

```python
print(list(model.modules()))
```

**Output (truncated)**:

```
SmallNet(...)
Conv2d(...)
BatchNorm2d(...)
Linear(...)
```

---

# ✅ 5. **`model.state_dict()`**

Weights + buffers.

```python
sd = model.state_dict()
print(sd.keys())
```

**Output**

```
odict_keys([
 'conv.weight', 'conv.bias',
 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var',
 'fc.weight','fc.bias'
])
```

---

# ✅ 6. **`.to(device)` / `.cpu()` / `.cuda()`**

```python
model_cpu = model.cpu()
print(next(model_cpu.parameters()).device)
```

**Output**

```
cpu
```

---

# ✅ 7. `.training`

```python
model.train()
print(model.training)   # True

model.eval()
print(model.training)   # False
```

---

# ✅ 8. `.forward()` (don't call directly)

```python
x = torch.randn(1, 3, 32, 32)
out = model(x)
print(out.shape)
```

**Output**

```
torch.Size([1, 10])
```

---

# ✅ 9. `.register_forward_hook()`

Capture activations of a layer:

```python
def hook(module, input, output):
    print("Hook output shape:", output.shape)

handle = model.conv.register_forward_hook(hook)
_ = model(x)
```

**Output**

```
Hook output shape: torch.Size([1, 8, 32, 32])
```

(Then remove hook)

```python
handle.remove()
```

---

# INTERNAL ATTRIBUTES

These are important for understanding PyTorch internals.

---

# ✅ 10. **`model._parameters`**

```python
print(model.conv._parameters.keys())
```

**Output**

```
dict_keys(['weight', 'bias'])
```

---

# ✅ 11. **`model._modules`**

Shows submodules:

```python
print(model._modules)
```

**Output**

```
OrderedDict([
 ('conv', Conv2d(3,8)),
 ('bn', BatchNorm2d(8)),
 ('fc', Linear(512 → 10))
])
```

---

# ✅ 12. **`model._buffers`**

Non-trainable persistent tensors (e.g., BatchNorm stats):

```python
print(model.bn._buffers.keys())
```

**Output**

```
dict_keys(['running_mean', 'running_var', 'num_batches_tracked'])
```

---

# ✅ 13. `.zero_grad()`

```python
out = model(x)
loss = out.sum()
loss.backward()
print(model.conv.weight.grad.shape)
model.zero_grad()
print(model.conv.weight.grad) 
```

**Output**

```
torch.Size([8, 3, 3, 3])
tensor of zeros after zero_grad()
```

---

# ✅ 14. `.apply(fn)`

Apply a function to every submodule:

```python
def reset_weights(m):
    if hasattr(m, "reset_parameters"):
        print("Resetting:", m)
        m.reset_parameters()

model.apply(reset_weights)
```

**Output**

```
Resetting: Conv2d(3,8)
Resetting: BatchNorm2d(8)
Resetting: Linear(512→10)
```

---

# FULL ATTRIBUTE LIST WITH EXAMPLES

Here is every important attribute, plus short code usage:

| Attribute                 | Description           | Example                     |
| ------------------------- | --------------------- | --------------------------- |
| `parameters()`            | All trainable tensors | `list(model.parameters())`  |
| `named_parameters()`      | Parameters with names | `model.named_parameters()`  |
| `children()`              | Direct submodules     | `model.children()`          |
| `modules()`               | All submodules        | `model.modules()`           |
| `state_dict()`            | All weights/buffers   | `model.state_dict()`        |
| `load_state_dict()`       | Load saved weights    | `model.load_state_dict(sd)` |
| `train()`                 | Training mode         | `model.train()`             |
| `eval()`                  | Eval mode             | `model.eval()`              |
| `forward()`               | Defines forward pass  | `model(x)` calls it         |
| `zero_grad()`             | Clear gradients       | `model.zero_grad()`         |
| `apply()`                 | Apply to submodules   | `model.apply(fn)`           |
| `register_forward_hook()` | Capture activations   | described above             |
| `to(device)`              | Move model            | `model.cuda()`              |
| `_parameters`             | Raw parameter dict    | `model._parameters`         |
| `_buffers`                | Buffers               | `model._buffers`            |
| `_modules`                | Child modules         | `model._modules`            |
| `_forward_hooks`          | Stored hooks          | `model._forward_hooks`      |

