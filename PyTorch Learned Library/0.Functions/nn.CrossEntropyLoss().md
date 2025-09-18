

## Overview

`nn.CrossEntropyLoss` is a loss function in PyTorch designed for multi-class classification problems. It combines two operations:

1. **Softmax**: Converts raw logits (unnormalized predictions) into probabilities.
2. **Negative Log-Likelihood (NLL) Loss**: Measures the difference between the predicted probabilities and the true class labels.

This combination ensures numerical stability and efficiency during training.

---

## Mathematical Formulation

For a single sample with $C$ classes:

$$
\text{CrossEntropyLoss}(x, y) = - \log \left( \frac{e^{x_y}}{\sum_{i=1}^{C} e^{x_i}} \right)
$$

Where:

* $x$ is the vector of raw logits (size $C$).
* $x_y$ is the logit corresponding to the true class $y$.
* The denominator is the sum of exponentials of all logits, ensuring the output is a valid probability distribution.

This formula computes the negative log probability of the true class, penalizing the model more when it's confident about an incorrect prediction.

---

## Parameters

* **`weight`**: A manual rescaling weight given to each class. This can be useful for handling class imbalance.
* **`size_average`**: Deprecated. Use `reduction` instead.
* **`ignore_index`**: Specifies a target value that is ignored and does not contribute to the input gradient.
* **`reduce`**: Deprecated. Use `reduction` instead.
* **`reduction`**: Specifies the method to reduce the loss. Options are:

  * `'none'`: No reduction will be applied.
  * `'mean'`: The sum of the output will be divided by the number of elements in the output.
  * `'sum'`: The output will be summed.
* **`label_smoothing`**: Applies label smoothing, a regularization technique that softens the target labels to make the model less confident.

---

## Example Usage

```python
import torch
import torch.nn as nn

# Sample logits (raw predictions) for 3 classes
logits = torch.tensor([[1.0, 2.0, 0.5], [1.2, 0.9, 1.5]])
# True class labels
labels = torch.tensor([1, 2])

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Compute the loss
loss = loss_fn(logits, labels)
print(f"Cross-Entropy Loss: {loss.item()}")
```

In this example, `logits` are the raw outputs from the model, and `labels` are the true class indices. The loss function computes the cross-entropy loss for each sample and averages them.

---

## Key Insights

* **Logits as Input**: Always provide raw logits (not probabilities) to `nn.CrossEntropyLoss`. This is because the function internally applies the softmax operation. Providing probabilities would lead to incorrect results.
* **Class Indices**: The target labels should be provided as class indices (not one-hot encoded vectors). For example, for 3 classes, valid labels are 0, 1, or 2.
* **Numerical Stability**: By combining softmax and NLL loss, PyTorch ensures numerical stability, especially when dealing with very small or large logits.



