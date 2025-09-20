

# 🔹 What is `torch.optim.SGD`?

`torch.optim.SGD` implements **Stochastic Gradient Descent (SGD)**, a fundamental optimization algorithm used to train neural networks.

* **Stochastic** → instead of computing gradients on the whole dataset (which is slow), it updates weights based on **mini-batches** of data.
* **Gradient Descent** → adjusts model parameters in the direction that reduces the loss, guided by the gradient.

In PyTorch, `SGD` can also include:

* **Momentum** → to smooth updates and accelerate convergence.
* **Nesterov momentum** → a more advanced momentum variant.
* **Weight decay** → acts like L2 regularization to prevent overfitting.

---

# 🔹 Formula

The basic update rule for a parameter $\theta$ is:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta} L(\theta_t)
$$

Where:

* $\eta$ = learning rate
* $\nabla_{\theta} L(\theta_t)$ = gradient of the loss w\.r.t parameter

### With **Momentum**:

$$
v_{t+1} = \mu v_t + \nabla_{\theta} L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

Where:

* $v_t$ = velocity (running average of past gradients)
* $\mu$ = momentum factor (0–1, usually 0.9)

Momentum helps escape local minima and smooths noisy updates.

---

# 🔹 Key Parameters

```python
torch.optim.SGD(
    params,                # model parameters
    lr=0.01,               # learning rate
    momentum=0,            # momentum factor (0 means plain SGD)
    dampening=0,           # reduces the effect of momentum
    weight_decay=0,        # L2 penalty (regularization)
    nesterov=False         # enable Nesterov momentum
)
```

* **lr**: Step size. Too large = unstable training; too small = slow training.
* **momentum**: Helps accelerate convergence (commonly 0.9).
* **dampening**: Reduces the influence of momentum (rarely used, default = 0).
* **weight\_decay**: Adds L2 regularization, common values: `1e-4` or `1e-5`.
* **nesterov**: If `True`, uses Nesterov accelerated gradient (more accurate look-ahead update).

---

# 🔹 Usage Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example model
model = nn.Linear(10, 1)

# Loss function
criterion = nn.MSELoss()

# SGD optimizer with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Dummy training loop
for epoch in range(5):
    inputs = torch.randn(32, 10)   # batch of 32 samples
    targets = torch.randn(32, 1)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward + Update
    optimizer.zero_grad()   # reset gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update parameters

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

---

# 🔹 When to Use SGD

✅ Good when:

* You want a **baseline optimizer** (many deep learning papers start with SGD).
* You need **control over regularization** (momentum, weight decay, etc.).
* You’re working with large datasets (mini-batch updates).

⚠️ Limitations:

* Slower to converge compared to **Adam** or **RMSprop**.
* More sensitive to learning rate choices.

---

# 🔹 Summary

* **SGD** = core optimizer in PyTorch for training neural nets.
* Works by updating parameters based on gradient of the loss.
* Can be extended with **momentum**, **Nesterov**, and **weight decay**.
* Simple but powerful; often used as a baseline before switching to adaptive optimizers like Adam.

