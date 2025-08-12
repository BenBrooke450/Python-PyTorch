

## What is `nn.Module` in PyTorch?

`nn.Module` is a **base class** for all neural network components in PyTorch. Whenever you want to create a neural network model or even a building block (like a layer, a block of layers, or an entire model), you typically **subclass `nn.Module`**.

---

### Why is `nn.Module` important?

* It provides a **standard structure** to define layers and computations.
* It **automatically handles parameters** (weights and biases) for you.
* It enables easy **saving and loading** of models.
* It supports **moving models to different devices** (CPU, GPU).
* It manages the **forward pass logic** clearly.
* It integrates seamlessly with other PyTorch features (like optimizers, loss functions, etc.).

---

### Anatomy of `nn.Module`

When you create your own model, you usually:

1. **Subclass `nn.Module`.**
2. Define layers as class attributes in the `__init__()` method.
3. Define the forward computation in the `forward()` method.
4. Instantiate the model and call it with input data (the call invokes `forward` under the hood).

---

### Example of a simple custom neural network using `nn.Module`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define layers here
        self.fc1 = nn.Linear(in_features=10, out_features=50)  # fully connected layer 1
        self.fc2 = nn.Linear(50, 1)  # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply fc1 and ReLU activation
        x = self.fc2(x)          # Apply output layer
        return x

model = SimpleNN()
```

---

### How `nn.Module` works internally:

* **Storing parameters:**
  When you assign layers like `self.fc1 = nn.Linear(...)` inside `__init__`, the `nn.Module` base class keeps track of all the parameters (weights and biases) in those layers. You don’t have to manually gather them.

* **Forward method:**
  The method `forward()` defines how input data flows through the layers to produce output. You **never call `forward` directly**; instead, you call the model instance like a function, e.g. `output = model(input)`. This internally calls `forward()`.

* **Parameter access:**
  You can access all model parameters easily with `model.parameters()` or `model.named_parameters()`. This is used by optimizers (e.g., `torch.optim.Adam`) to update weights.

* **Device management:**
  Calling `model.to(device)` moves all parameters to the specified device (CPU or GPU) seamlessly.

* **Saving and loading:**
  You can save the entire model or just the parameters using `torch.save(model.state_dict())` and load them later. `nn.Module` ensures consistent structure.

---

### Summary Table of `nn.Module` Features

| Feature                  | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| **Subclassable**         | Create custom models by subclassing `nn.Module`        |
| **Parameter tracking**   | Automatically tracks all learnable parameters          |
| **Forward method**       | Define computation logic in `forward()` method         |
| **Callable**             | Call the model instance directly, it calls `forward()` |
| **Parameter access**     | Easily retrieve parameters for optimization            |
| **Device support**       | Move entire model to CPU or GPU with `.to(device)`     |
| **Model saving/loading** | Save/load model state dictionaries                     |
| **Nesting**              | Modules can contain other modules as submodules        |

---

### Why not just use functions?

* Functions alone can't store **parameters** (weights).
* `nn.Module` encapsulates **state (parameters)** and **behavior (forward pass)**.
* Enables complex architectures with reusable submodules.

---

### Recap

| Concept               | Explanation                                                 |
| --------------------- | ----------------------------------------------------------- |
| `nn.Module`           | Base class for all PyTorch neural network models and layers |
| `__init__()`          | Define layers and submodules                                |
| `forward()`           | Define data flow and computation                            |
| `model(input)`        | Calls `forward` under the hood                              |
| `model.parameters()`  | Access learnable parameters                                 |
| `model.to(device)`    | Move model to GPU or CPU                                    |
| `torch.save()`/`load` | Save and load model weights/state                           |

---


<br><br><br><br><br>

# Example 1



### Example: A simple feedforward neural network for regression

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()  # Initialize base nn.Module

        # Define layers here:
        # A fully connected layer taking 10 input features and outputting 20 features
        self.fc1 = nn.Linear(in_features=10, out_features=20)

        # Another fully connected layer reducing 20 features to 1 output (regression)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        # x: input tensor of shape [batch_size, 10]

        x = self.fc1(x)          # Pass input through first linear layer
        x = F.relu(x)            # Apply ReLU activation function (non-linearity)
        x = self.fc2(x)          # Pass through second linear layer (output layer)

        return x                 # Return output tensor of shape [batch_size, 1]


# Create model instance
model = SimpleNN()

# Example input: batch of 5 samples, each with 10 features
input_tensor = torch.randn(5, 10)  # random data

# Forward pass: get model prediction
output = model(input_tensor)

print(output)
```

---

### Detailed explanation of each part:

| Code Part                           | Explanation                                                                             |
| ----------------------------------- | --------------------------------------------------------------------------------------- |
| `import torch`                      | Imports the PyTorch library                                                             |
| `import torch.nn as nn`             | Imports PyTorch's neural network module with layers and utilities                       |
| `import torch.nn.functional as F`   | Imports functional interface (activations, loss functions, etc.)                        |
| `class SimpleNN(nn.Module):`        | Define a new class `SimpleNN` that inherits from PyTorch's `nn.Module` base class       |
| `def __init__(self):`               | Constructor method where we define layers and submodules                                |
| `super(SimpleNN, self).__init__()`  | Calls the constructor of the parent `nn.Module` class to properly initialize the module |
| `self.fc1 = nn.Linear(10, 20)`      | Defines a fully connected (linear) layer that maps 10 inputs to 20 outputs              |
| `self.fc2 = nn.Linear(20, 1)`       | Defines another linear layer that maps 20 inputs to 1 output (like a regression target) |
| `def forward(self, x):`             | Defines the forward pass: the data flow logic of the network                            |
| `x = self.fc1(x)`                   | Applies the first linear transformation to input tensor `x`                             |
| `x = F.relu(x)`                     | Applies ReLU activation function for non-linearity (zero out negatives, keep positives) |
| `x = self.fc2(x)`                   | Applies the second linear layer to get output                                           |
| `return x`                          | Returns the output tensor                                                               |
| `model = SimpleNN()`                | Creates an instance of the model                                                        |
| `input_tensor = torch.randn(5, 10)` | Creates a random tensor with shape (batch\_size=5, features=10) to simulate input data  |
| `output = model(input_tensor)`      | Calls the model on input, which internally calls `forward()`                            |
| `print(output)`                     | Prints the output tensor, shape (5, 1), predicted values for each input in the batch    |

---

### Key points to remember:

* **`nn.Module` base class:** Enables the model to track layers and parameters.
* **Layers as attributes:** `self.fc1` and `self.fc2` are learnable layers automatically tracked.
* **Forward method:** Defines how input flows through the layers.
* **Activation:** Non-linear function (`ReLU`) applied after first layer for model complexity.
* **Instantiation:** `model = SimpleNN()` creates the model object.
* **Calling the model:** `model(input_tensor)` runs the forward pass and returns predictions.


<br><br><br><br>


You’re asking if you **have to manually call activation functions like `F.relu` separately after each layer**, or if there’s a way to combine layers + activations so the whole thing runs automatically in sequence.

---

### Quick answer:

* In the example, yes, you explicitly call the activation function after the layer inside the `forward()` method.
* **But you can combine layers and activations into a single "module" using `nn.Sequential` so they run automatically one after another.**

---

### What is `nn.Sequential`?

`nn.Sequential` is a **container module** that chains together layers and functions in order. When you call the whole sequential model on input, it automatically passes the input through each layer/function in sequence.

---

### Example: Using `nn.Sequential` to combine layers and activations

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Chain fc1 layer and ReLU activation together
        self.layer1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        # Output layer (no activation)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)   # Automatically does Linear -> ReLU
        x = self.fc2(x)      # Final output layer
        return x

model = SimpleNN()
input_tensor = torch.randn(5, 10)
output = model(input_tensor)
print(output)
```

---

### How this helps:

* You don’t need to manually call activation functions inside `forward` when using `nn.Sequential`.
* You can chain **any number** of layers and activations.
* Code becomes cleaner and modular.

---

### Summary:

| Approach              | Where activations are called                    | Pros                                                 | Cons                         |
| --------------------- | ----------------------------------------------- | ---------------------------------------------------- | ---------------------------- |
| Manual in `forward()` | You call activation functions explicitly        | More flexible for complex architectures              | More verbose                 |
| `nn.Sequential`       | Layers and activations chained inside container | Cleaner, automatic execution of layers + activations | Less flexible for some cases |



