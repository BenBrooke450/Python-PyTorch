Perfect 👍 let’s break down **`torchmetrics.Accuracy`** so you’ve got a clean summary for future reference.

---

#  `torchmetrics.Accuracy` Summary

## 1. What it is

`Accuracy` is a metric from [torchmetrics](https://torchmetrics.readthedocs.io/) that computes how many predictions match the ground truth labels. It’s widely used for **classification tasks** (binary, multiclass, multilabel).

---

## 2. Required arguments

When creating the metric, you **must** specify the task type:

```python
from torchmetrics import Accuracy

# Binary classification
acc_binary = Accuracy(task="binary")

# Multiclass classification (needs num_classes)
acc_multi = Accuracy(task="multiclass", num_classes=3)

# Multilabel classification
acc_multilabel = Accuracy(task="multilabel", num_labels=5)
```

---

## 3. Inputs

* **Binary:** predictions can be probabilities/logits (`float`) or class labels (`0/1`).
* **Multiclass:** predictions should be class indices (`[0, 1, ..., num_classes-1]`) or logits that you `argmax`.
* **Multilabel:** predictions are per-label probabilities or binary outputs.

---

## 4. Typical usage

```python
import torch
from torchmetrics import Accuracy

# Example multiclass
y_pred = torch.tensor([[0.1, 0.2, 0.7],
                       [0.8, 0.1, 0.1],
                       [0.2, 0.3, 0.5]])
y_true = torch.tensor([2, 0, 2])

acc = Accuracy(task="multiclass", num_classes=3)

# Convert logits to predicted class
pred_classes = torch.argmax(y_pred, dim=1)

print(acc(pred_classes, y_true))  # tensor(1.) → 100%
```

---

## 5. Works with batches

`torchmetrics` metrics are **stateful** — you can keep updating them across mini-batches and compute the final score at the end:

```python
acc = Accuracy(task="multiclass", num_classes=10)

for batch_x, batch_y in dataloader:
    preds = model(batch_x)
    preds = preds.argmax(dim=1)
    acc.update(preds, batch_y)

final_acc = acc.compute()
print(final_acc)
```

* `update(preds, target)` → accumulates batch results
* `compute()` → returns the aggregated accuracy
* `reset()` → clears stored states

---

## 6. Device compatibility

`torchmetrics` automatically works on **CPU or GPU**, just move the metric to the device:

```python
acc = Accuracy(task="multiclass", num_classes=10).to("cuda")
```

---

 **Summary in one line:**
`torchmetrics.Accuracy` is a flexible, GPU-friendly metric for binary, multiclass, and multilabel classification, where you must specify `task` (and `num_classes`/`num_labels` if needed).



Great question 🙌 — let’s go into the **mathematics** behind accuracy in `torchmetrics`.

---

<br><br><br><br>


#  The Mathematics of Accuracy

At its core, **accuracy** is:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

---

## 1. **Binary Classification** (task="binary")

* Predictions are either $0$ or $1$.
* Given targets $y_i \in \{0, 1\}$ and predictions $\hat{y}_i \in \{0, 1\}$:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{y_i = \hat{y}_i\}
$$

where $\mathbf{1}\{condition\} = 1$ if condition is true, else 0.

 Example:
Targets = \[1, 0, 1, 1]
Preds   = \[1, 0, 0, 1]

Correct = 3 / 4 = **0.75**

---

## 2. **Multiclass Classification** (task="multiclass")

* Each sample belongs to exactly **one** of $C$ classes.
* Predictions are usually logits → you take `argmax` to pick the class.

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{y_i = \arg\max(\hat{y}_i)\}
$$

where $\hat{y}_i$ is the prediction vector for sample $i$.

 Example (3 classes):
Targets = \[2, 0, 1]
Predictions = \[\[0.1, 0.3, 0.6], \[0.8, 0.1, 0.1], \[0.2, 0.7, 0.1]]
Argmax preds = \[2, 0, 1] → all correct → Accuracy = **1.0**

---

## 3. **Multilabel Classification** (task="multilabel")

* Each sample can belong to **multiple classes simultaneously**.
* Predictions are usually per-label probabilities → threshold (e.g. ≥ 0.5) → convert to 0/1.

For each sample $i$ and label $j$:

$$
\text{Accuracy} = \frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{j=1}^{L} \mathbf{1}\{y_{ij} = \hat{y}_{ij}\}
$$

where $L$ = number of labels.

 Example (2 samples, 3 labels):
Targets = \[\[1,0,1], \[0,1,1]]
Preds   = \[\[1,0,0], \[0,1,1]]

Correct matches = 5 / 6 = **0.833**

