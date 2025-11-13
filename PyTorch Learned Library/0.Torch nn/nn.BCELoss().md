

# `torch.nn.BCELoss`

###  What it is

* **Binary Cross-Entropy Loss** (BCE) is used for **binary classification tasks**.
* It compares the predicted probabilities (after a **sigmoid activation**) with the true binary labels (0 or 1).
* It penalizes predictions that are far from the true labels, with the penalty growing logarithmically.

 **Important:** Unlike `BCEWithLogitsLoss`, `BCELoss` **expects probabilities** (values between 0 and 1) as input, not raw logits.

---

###  Formula

Given predictions **ŷᵢ** (probabilities) and binary labels **yᵢ ∈ {0,1}**, with n samples:

**L(y, ŷ) = − (1/n) · Σ \[ yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ) ]**

* If **yᵢ = 1** → loss = −log(ŷᵢ)
  (high penalty if predicted probability for class 1 is small)
* If **yᵢ = 0** → loss = −log(1 − ŷᵢ)
  (high penalty if predicted probability for class 0 is small)

---

###  Arguments

```python
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
```

* **`weight`**: Optional tensor assigning a weight to each sample. Helps with class imbalance.
* **`reduction`**:

  * `'mean'` → average the loss across all samples (default).
  * `'sum'` → sum of all losses.
  * `'none'` → return individual losses per sample.

---

### When to use it

* **Binary classification problems** where model outputs are probabilities (after `sigmoid`).
* **Multi-label classification** (treating each label as independent binary classification).

---

### Example Usage

```python
import torch
from torch import nn

loss_fn = nn.BCELoss()

# Model outputs after sigmoid
y_pred = torch.tensor([[0.9], [0.2], [0.7]])
# True labels
y_true = torch.tensor([[1.], [0.], [1.]])

loss = loss_fn(y_pred, y_true)
print(loss.item())
```

---

### Detailed Explanation of Each Part

1. **`yᵢ·log(ŷᵢ)`**

   * Encourages the model to predict probabilities close to 1 when the true label is 1.
   * If ŷᵢ is small while yᵢ=1, log(ŷᵢ) is very negative → big penalty.

2. **`(1 − yᵢ)·log(1 − ŷᵢ)`**

   * Encourages the model to predict probabilities close to 0 when the true label is 0.
   * If ŷᵢ is close to 1 while yᵢ=0, log(1 − ŷᵢ) is very negative → big penalty.

3. **Negative sign (−)**

   * Turns the log-likelihood (which we want to maximize) into a loss (which we minimize).

4. **Averaging (1/n)**

   * Normalizes the loss across the batch so that it doesn’t scale with batch size.

---

### Comparison: `BCELoss` vs `BCEWithLogitsLoss`

* **`BCELoss`**:

  * Requires you to **apply sigmoid manually** to model outputs before passing them in.
  * Less numerically stable (log(0) issues if predictions saturate).

* **`BCEWithLogitsLoss`**:

  * Takes raw logits (no sigmoid).
  * More numerically stable (combines sigmoid + BCE in one step).
  * Preferred in almost all practical deep learning setups.

---

### Typical Systems

* Logistic regression (binary classification).
* Simple neural networks for binary outcomes.
* Multi-label classification tasks (e.g. multi-tag text classification).

---

### Common Pitfalls

1. **Forgetting to apply sigmoid**

   * If you pass raw logits into `BCELoss`, it will treat them as probabilities, leading to nonsense losses.
   * Use `torch.sigmoid()` first.

2. **Wrong target shape/type**

   * Targets must be floats (`0.` or `1.`), not integers.
   * Shapes of `y_pred` and `y_true` must match.

3. **Numerical instability**

   * If `ŷᵢ` is 0 or 1 exactly, `log(ŷᵢ)` or `log(1−ŷᵢ)` will be undefined.
   * PyTorch handles some clipping internally, but this is why `BCEWithLogitsLoss` is safer.

---

**In summary:**
`torch.nn.BCELoss` implements the binary cross-entropy loss for probabilities. It’s conceptually simple but less numerically stable than `BCEWithLogitsLoss`. Use it if you explicitly want to apply `sigmoid` yourself, but in most cases, **`BCEWithLogitsLoss` is preferred**.



<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>


# ✅ `nn.BCEWithLogitsLoss`

vs

# ❌ `nn.BCELoss`

And when to use which.

---

# **1. BCEWithLogitsLoss — the *correct* way (recommended)**

`BCEWithLogitsLoss` **expects raw logits** and internally applies:

```
Sigmoid + BCE
```

Advantages:

* more numerically stable
* no need for `sigmoid()` in your model
* prevents exploding/vanishing gradients

---

# CODE USING `BCEWithLogitsLoss`

### **Model**

**Do NOT use sigmoid in the forward pass.**

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)    # raw logits
```

### **Loss**

```python
loss_fn = nn.BCEWithLogitsLoss()
```

### **Training step**

```python
logits = model(X)                # shape: [batch, 1]
loss = loss_fn(logits, y.float())
```

### **Prediction**

Apply sigmoid at evaluation time:

```python
probs = torch.sigmoid(logits)
preds = (probs > 0.5).int()
```

---

# **2. BCELoss — old method (not recommended)**

`BCELoss` **expects probabilities**, so you MUST apply `sigmoid()` in the model or training loop.

Problems:

* unstable
* NaNs if prob = 0 or 1
* worse training behavior

---

# CODE USING `BCELoss`

### **Model**

**You MUST include sigmoid inside forward()**

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # probability
```

### **Loss**

```python
loss_fn = nn.BCELoss()
```

### **Training step**

```python
probs = model(X)
loss = loss_fn(probs, y.float())
```

### **Prediction**

Already probabilities, so:

```python
preds = (probs > 0.5).int()
```

---

# Side-by-side comparison (TL;DR)

| Task                | BCEWithLogitsLoss        | BCELoss                              |
| ------------------- | ------------------------ | ------------------------------------ |
| Model output        | **logits** (no sigmoid)  | **probabilities** (sigmoid required) |
| Where sigmoid?      | **Inside loss function** | **Inside model**                     |
| Numerical stability | ⭐ Best                   | ❌ Worse                              |
| Recommended?        | **YES**                  | Only for special cases               |

---

# When to use which?

* **Binary classification?** → `BCEWithLogitsLoss`
* **Multi-label classification?** (e.g., 8 independent labels) → `BCEWithLogitsLoss`
* **You really need probabilities inside forward()?** → `BCELoss`
