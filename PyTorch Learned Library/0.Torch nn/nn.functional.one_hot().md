
# 1. What `one_hot()` Does

`torch.nn.functional.one_hot()` converts integer class labels into **one-hot vectors**.

A one-hot vector has:

* 1 at the position of the class index
* 0 everywhere else

Example:

```
Class index = 2
num_classes = 4

One-hot → [0, 0, 1, 0]
```

---

# 2. Function Definition

```python
torch.nn.functional.one_hot(tensor, num_classes=None)
```

* `tensor` = class indices (dtype must be `torch.long`)
* `num_classes` = total number of classes (optional)

---

# 3. Basic Example (1D input)

```python
import torch
import torch.nn.functional as F

labels = torch.tensor([0, 2, 1])
one_hot = F.one_hot(labels, num_classes=3)

print(one_hot)
```

Output:

```
tensor([[1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]])
```

Explanation:

* Class `0` → `[1, 0, 0]`
* Class `2` → `[0, 0, 1]`
* Class `1` → `[0, 1, 0]`

Shape of output: `[3, 3]`

---

# 4. Scalar Example (0D input)

```python
label = torch.tensor(3)
one_hot = F.one_hot(label, num_classes=5)
print(one_hot)
```

Output:

```
tensor([0, 0, 0, 1, 0])
```

---

# 5. Multi-dimensional Example

If your labels are 2D:

```python
labels = torch.tensor([[1, 0],
                       [2, 1]])

one_hot = F.one_hot(labels, num_classes=3)
print(one_hot)
```

Output shape: `[2, 2, 3]`
Output:

```
tensor([[[0, 1, 0],
         [1, 0, 0]],

        [[0, 0, 1],
         [0, 1, 0]]])
```

Each number is converted to a one-hot vector in place.

---

# 6. Using one_hot() with Model Predictions

Assume your model outputs logits:

```python
logits = model(x)   # shape [batch, num_classes]
_, preds = torch.max(logits, dim=1)
```

Predicted classes:

```
tensor([2, 0, 1])
```

Convert to one-hot:

```python
one_hot_preds = F.one_hot(preds, num_classes=3)
```

Output:

```
tensor([[0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])
```

---

# 7. Converting Labels Before a BCE Loss

If your problem is **multi-label**, you need one-hot labels:

```python
labels = torch.tensor([1, 0, 2])
labels_one_hot = F.one_hot(labels, num_classes=3).float()
```

For a BCE loss:

```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, labels_one_hot)
```

(Don’t do this with CrossEntropyLoss — that requires class indices, not one-hot.)

---

# 8. Common Errors and Fixes

### Error 1:

`RuntimeError: expected dtype Long but got Float`

Fix:

```python
labels = labels.long()
```

### Error 2:

Using one-hot labels with `nn.CrossEntropyLoss`
This is incorrect.

Correct usage:

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)     # labels are class indices, not one-hot
```

---

# 9. Summary Table

| Concept             | Explanation                          |
| ------------------- | ------------------------------------ |
| Input               | integer class labels                 |
| Output              | one-hot encoded vectors              |
| Required dtype      | `torch.long`                         |
| Output shape        | original shape + one extra dimension |
| Not compatible with | `CrossEntropyLoss`                   |
| Commonly used for   | BCE losses, metrics, visualisation   |

---

If you want, I can also produce:

* a one-hot confusion matrix example
* a visualization that highlights the 1 positions
* a method to convert one-hot back to class indices

Just tell me.
