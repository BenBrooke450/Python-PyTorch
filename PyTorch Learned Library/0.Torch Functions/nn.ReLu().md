
# **ReLU (Rectified Linear Unit) — Summary**

## Definition

The **ReLU activation function** is defined as:

Absolutely! Here’s a **Markdown-ready version** of your ReLU description that you can copy and paste into a `.md` file:

ReLU(x) = max(0, x)

- If x > 0 → output = x  
- If x ≤ 0 → output = 0


---

### Example in Markdown

---

This will render properly in **Jupyter Notebook, GitHub, or any Markdown renderer with LaTeX support**.

If you want, I can rewrite your **entire ReLU summary** in the same Markdown-friendly style for easy copy-paste. Do you want me to do that?


It is **piecewise linear**, simple, and widely used in neural networks.

---

## Why we use ReLU

* Introduces **non-linearity** → allows neural networks to learn complex functions
* Computationally simple → fast to compute
* Helps reduce **vanishing gradient problem** compared to sigmoid/tanh
* Sparse activations → negative values become 0, which can improve efficiency

---

## Usage in PyTorch

PyTorch provides **two main ways** to apply ReLU:

### ✅ Using `nn.ReLU` (module)

```python
import torch
import torch.nn as nn

relu = nn.ReLU()

x = torch.tensor([-2.0, 0.0, 3.5])
y = relu(x)
print(y)  # tensor([0.0, 0.0, 3.5])
```

### ✅ Using `torch.nn.functional.relu` (functional)

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, 0.0, 3.5])
y = F.relu(x)
print(y)  # tensor([0.0, 0.0, 3.5])
```

---

## Typical use in a neural network

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU after first layer
        x = self.fc2(x)          # Output layer (no ReLU)
        return x
```

* **Hidden layers** → ReLU activation
* **Output layer** → typically no ReLU (depends on task, e.g., sigmoid for binary classification)

---

## Key notes

| Feature      | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| Formula      | (\text{ReLU}(x) = \max(0, x))                                               |
| Pros         | Simple, non-linear, avoids vanishing gradient, sparse activations           |
| Cons         | “Dying ReLU” problem → neurons can get stuck at 0 if learning rate too high |
| Alternatives | LeakyReLU, ELU, GELU                                                        |

---

## Quick example

```python
x = torch.tensor([-5, -1, 0, 2, 5], dtype=torch.float32)
y = F.relu(x)
print(y)  # tensor([0., 0., 0., 2., 5.])
```

Output is **all negative values zeroed**, positive values unchanged.

