

# 🔹 `torch.save()` in PyTorch

## 🔸 What it does

* Saves a PyTorch object to disk.

* Often used to save:

  * **Model parameters (state\_dict)**
  * **Entire models**
  * **Tensors**
  * **Dictionaries of training checkpoints**

* It uses **Python’s `pickle` module** under the hood, so most Python objects are serializable.

---

## 🔸 Syntax

```python
torch.save(obj, f)
```

* `obj` → The object you want to save (e.g., model.state\_dict(), tensor, dict).
* `f` → A file name (`'model.pth'`) or file-like object (opened with `open()` in binary mode).

---

## 🔸 Common Use Cases

### 1. Save a Tensor

```python
import torch

x = torch.randn(3, 3)
torch.save(x, "tensor.pth")

# later...
x_loaded = torch.load("tensor.pth")
print(x_loaded)
```

### 2. Save Model Parameters (Best Practice ✅)

```python
torch.save(model.state_dict(), "model.pth")
```

Later, you can reload them into the same model architecture:

```python
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
```

### 3. Save Entire Model (Not Recommended ❌)

```python
torch.save(model, "entire_model.pth")
model = torch.load("entire_model.pth")
```

* Works, but less portable because it depends on the exact class definition being available.

### 4. Save Training Checkpoints

```python
checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "loss": loss
}
torch.save(checkpoint, "checkpoint.pth")
```

---

## 🔸 Why It’s Useful

* Lets you **resume training** from a checkpoint.
* Lets you **deploy models** without retraining.
* Keeps experiments reproducible.

---

## 🔸 Key Notes

* Use `.pt` or `.pth` extensions by convention.
* Prefer saving `state_dict()` instead of the full model for portability.
* Always pair `torch.save()` with `torch.load()` to restore.

---

✅ **Summary:**
`torch.save()` is used to **serialize and save PyTorch objects (like tensors, model weights, or checkpoints) to disk**, usually in `.pt` or `.pth` format. Best practice is to save a model’s `state_dict()` and reload it later with `load_state_dict()` to ensure portability and reproducibility.

