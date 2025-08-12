

## **`torch.nn.Parameter`

### **1. Definition**

`torch.nn.Parameter` is a subclass of `torch.Tensor` that tells PyTorch:

> “This tensor is a **learnable parameter** of my model. Track it, compute gradients for it, and update it during training.”

When you assign an `nn.Parameter` to an attribute of an `nn.Module`, PyTorch **automatically registers** it so:

* It shows up in `model.parameters()`
* Optimizers can update it
* It participates in autograd

---

### **2. Why use `nn.Parameter`?**

* Built-in layers like `nn.Linear` or `nn.Conv2d` already create Parameters for you (weights, biases).
* You use `nn.Parameter` **when you want to create your own trainable tensors** in a custom layer or model.

If you just use a plain `torch.Tensor`, PyTorch won’t automatically track it unless you wrap it with `nn.Parameter`.

---

### **3. Key Characteristics**

| Feature                     | Description                                                                     |
| --------------------------- | ------------------------------------------------------------------------------- |
| **Subclass of Tensor**      | Behaves like a tensor but is treated as a model parameter                       |
| **requires\_grad=True**     | Default, so gradients are calculated during backprop                            |
| **Auto-registration**       | Stored in the model’s internal parameter list if assigned as a module attribute |
| **For learnable variables** | Use for weights, biases, embeddings, etc., that you want to train               |

---

### **4. Basic Example — Custom Learnable Parameter**

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a custom 3x3 learnable parameter
        self.my_weight = nn.Parameter(torch.randn(3, 3))

    def forward(self, x):
        return x @ self.my_weight  # matrix multiplication

model = MyModel()

# All trainable parameters (includes my_weight)
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
```

**Output:**

```
my_weight torch.Size([3, 3]) True
```

✅ `my_weight` is trainable and tracked automatically.

---

### **5. Example — Why a Normal Tensor Won’t Work**

```python
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_tensor = torch.randn(3, 3)  # Not a Parameter

bad_model = BadModel()
print(list(bad_model.parameters()))  # []
```

❌ `my_tensor` is ignored by `parameters()` because it’s not an `nn.Parameter`.

---

### **6. Using with an Optimizer**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(2, 3)   # input
loss = model(x).sum()   # fake loss
loss.backward()         # compute gradients
optimizer.step()        # update my_weight
```

Here, `my_weight` will be updated automatically during `optimizer.step()` because it’s in `model.parameters()`.

---

### **7. Summary Table**

| Term                    | Meaning                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| `torch.nn.Parameter`    | A special tensor marked as a learnable parameter                    |
| `model.parameters()`    | Returns all registered `nn.Parameter` objects in the model          |
| Default `requires_grad` | `True`                                                              |
| Best for                | Custom trainable weights/biases outside built-in layers             |
| Common mistake          | Using `torch.Tensor` instead of `nn.Parameter` for trainable values |

---

**In short:**
`nn.Parameter` is how you tell PyTorch,

> “This number is part of my model, please learn it.”
> Without it, the value won’t be optimized.



# Example 1