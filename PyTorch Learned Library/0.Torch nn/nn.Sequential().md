# `nn.Sequential()` — Detailed Summary

`nn.Sequential` is a **container module** in PyTorch that lets you build neural networks by **stacking layers in order**. It’s one of the simplest and fastest ways to define a model when the data flows **strictly from layer 1 → layer 2 → layer 3 → ...** with no branching.

---

## 1. What `nn.Sequential` Does

It:

* Wraps a list of PyTorch layers or functions.
* Executes them **in the exact order** they are passed.
* Automatically registers all submodules so gradients flow correctly.
* Allows you to write compact models without manually coding a `forward()` method.

---

## 2. Basic Usage

### Example

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
```

This is equivalent to writing a custom `nn.Module` like:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

---

# 3. How Forward Pass Works

Input is passed **sequentially** through each layer:

```python
output = model(x)
# internally this becomes:
# x = layer1(x)
# x = layer2(x)
# x = layer3(x)
```

There is **no custom control over the forward method**; it is pre-defined by PyTorch.

---

# 4. Ways to Name Layers

### Unnamed layers (auto-incremented)

```python
nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU()
)
```

### Named layers (preferred for clarity)

```python
model = nn.Sequential(
    ('conv', nn.Conv2d(3, 16, 3)),
    ('relu', nn.ReLU())
)
```

### Using an `OrderedDict`

```python
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 20)),
    ('act', nn.ReLU()),
    ('fc2', nn.Linear(20, 1))
]))
```

---

# 5. Accessing Layers

```python
model[0]         # first layer
model[1]         # second layer
model.fc1        # only works if you named them
```

---

# 6. When `nn.Sequential` Is Useful

Use it when your model is:

* Feedforward
* Convolutional stack
* Multi-layer perceptron
* Simple CNN or simple encoder/decoder block
* Straight sequence without branching

Examples:

```python
nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
    nn.ReLU()
)
```

---

# 7. When **Not** to Use `nn.Sequential`

Avoid it when:

### 1. You need **multiple inputs** or **multiple outputs**

(e.g., attention mechanisms, additive/multiplicative branches)

### 2. You need **skip connections**

(ResNet-style `x + f(x)`)

### 3. You need **conditional logic** inside `forward()`

(if/else, loops dependent on data)

In these cases, write a custom `nn.Module`.

---

# 8. Common Pitfalls

### Pitfall 1: No internal variables in `forward()`

You cannot reuse intermediate results (e.g., for residual connections).

### Pitfall 2: Forgetting that each layer must take the output of the previous layer

Shapes must match exactly.

### Pitfall 3: No custom forward pass

You cannot insert debugging logic unless you wrap layers manually.

---

# 9. Practical Advanced Usage

### Sequential with functional layers

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU()
)
```

### Nested Sequential modules

```python
block = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

model = nn.Sequential(
    block,
    block,
    nn.Linear(64*32*32, 10)
)
```

---

# 10. TL;DR Summary

* `nn.Sequential` stacks modules in a fixed order.
* Great for simple feedforward models.
* No custom `forward()`; execution is automatic.
* Clean, compact, minimal code.
* Not suitable for complex architectures needing branching or custom logic.

