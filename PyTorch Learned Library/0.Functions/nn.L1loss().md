

# `torch.nn.L1Loss`

### What it is

* Also known as **Mean Absolute Error (MAE)**.
* A **regression loss function** that measures the average absolute difference between predictions and targets.
* It penalizes errors **linearly** (unlike MSE, which penalizes quadratically).

---

### Formula

Given predictions **ŷᵢ** and targets **yᵢ**, with n samples:

**L(y, ŷ) = (1/n) · Σ | yᵢ − ŷᵢ |**

* If `reduction='sum'`: L = Σ | yᵢ − ŷᵢ |
* If `reduction='mean'`: L = (1/n) Σ | yᵢ − ŷᵢ |
* If `reduction='none'`: returns a tensor of absolute differences for each sample.

---

### Arguments

```python
torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
```

* **`reduction`**:

  * `'mean'` → average over all samples (default).
  * `'sum'` → sum of all absolute errors.
  * `'none'` → return per-sample loss values.

---

### When to use it

* **Regression problems** where you care about **absolute deviations** rather than squared errors.
* When the dataset may contain **outliers** (MAE is more robust than MSE).
* When you want a model that is less sensitive to a few large errors.

---

### Example Usage

```python
import torch
from torch import nn

loss_fn = nn.L1Loss()

# Predictions and targets
y_pred = torch.tensor([2.5, 0.0, 2.0, 7.0])
y_true = torch.tensor([3.0, -0.5, 2.0, 8.0])

loss = loss_fn(y_pred, y_true)
print(loss.item())  # mean absolute error
```

* Output: `0.5` (since mean(|errors|) = (0.5 + 0.5 + 0 + 1) / 4)

---

### Comparison with MSE

* **MSE (L2 loss)**: Penalizes large errors more strongly → smoother gradients, useful in optimization.
* **MAE (L1 loss)**: Treats all errors equally → more robust to outliers but gradients are constant (less smooth).

👉 Many practitioners combine them (e.g., **Huber Loss**) to balance both properties.

---

### Typical Systems

* Regression neural networks.
* Robust regression with outliers.
* Autoencoders (when you care about absolute reconstruction errors).
* Image-to-image translation tasks (sometimes L1 is used as a perceptual loss).

---

### Common Pitfalls

1. **Slower convergence than MSE**

   * Because gradient is constant w\.r.t error (no larger push for bigger errors).
2. **Scaling matters**

   * If features/targets aren’t normalized, absolute errors may be misleading.
3. **Choice of reduction**

   * Default is `'mean'`. If you expect very small batch sizes, sometimes `'sum'` may be preferable.

---

✅ **In summary:**
`torch.nn.L1Loss` implements the **Mean Absolute Error**. It’s robust to outliers, simple, and often used in regression and reconstruction tasks. If you need smoother optimization but still robustness, consider **Huber Loss**.
