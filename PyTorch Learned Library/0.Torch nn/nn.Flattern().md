

# **PyTorch `nn.Flatten`**

## **1️⃣ What is `nn.Flatten`?**

* `nn.Flatten` is a **layer that converts multi-dimensional tensors into 2D tensors**: `(batch_size, features)`.
* Typically used **before fully connected (`nn.Linear`) layers** in CNNs.

**Example shape conversion:**

$$
\text{Input: } (N, C, H, W) \rightarrow \text{Output: } (N, C \times H \times W)
$$

Where:

* $N$ = batch size
* $C$ = channels (e.g., RGB = 3)
* $H, W$ = height and width of feature maps

---

## **2️⃣ Why Use `nn.Flatten`?**

1. Linear layers require **2D input**: `(batch_size, num_features)`
2. CNNs output **3D or 4D tensors**: `(N, C, H, W)`
3. Flattening converts each sample into a **single feature vector** while keeping batch dimension intact.

Without flattening → PyTorch throws a **shape mismatch error** when feeding into `nn.Linear`.

---

## **3️⃣ Realistic Example: CNN → Flatten → Fully Connected NN**

```python
import torch
import torch.nn as nn

# Example input: batch of 16 RGB images, size 32x32
x = torch.randn(16, 3, 32, 32)  # shape: (batch_size, channels, H, W)

# 1️⃣ Define a CNN with Flatten before fully connected layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # (16, 8, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 8, 16, 16)
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # (16, 16, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2)   # (16, 16, 8, 8)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()  # converts (16, 16, 8, 8) → (16, 1024)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(16*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # final output for 10 classes
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# 2️⃣ Instantiate model and forward pass
model = CNN()
output = model(x)
print("Output shape:", output.shape)
```

**Output:**

```
Output shape: torch.Size([16, 10])
```

---

### **Step-by-Step Explanation**

1. **Input:** `(16, 3, 32, 32)` → batch of 16 RGB images.
2. **Conv Layer 1:** `(16, 3, 32, 32)` → `(16, 8, 32, 32)`
3. **MaxPool 1:** `(16, 8, 32, 32)` → `(16, 8, 16, 16)`
4. **Conv Layer 2:** `(16, 8, 16, 16)` → `(16, 16, 16, 16)`
5. **MaxPool 2:** `(16, 16, 16, 16)` → `(16, 16, 8, 8)`
6. **Flatten:** `(16, 16, 8, 8)` → `(16, 1024)`
7. **FC Layer 1:** `(16, 1024)` → `(16, 128)`
8. **FC Layer 2:** `(16, 128)` → `(16, 10)` → final logits for 10 classes

---

## **4️⃣ Key Points About Flatten**

* Automatically keeps **batch dimension** (`N`) intact.
* Converts **spatial and channel dimensions** into a single vector per sample.
* Essential **bridge between CNN outputs and fully connected layers**.

---

## **5️⃣ What Happens If You Don’t Use Flatten?**

```python
x = model.conv_layers(x)  # shape: (16, 16, 8, 8)
output = model.fc_layers(x)  # ❌ ERROR
```

**Error:**

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

* Reason: `nn.Linear` expects `(batch_size, features)`, not `(batch_size, C, H, W)`.

---

✅ **Summary Table**

| Feature         | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| Purpose         | Flatten multi-dimensional tensor to 2D `(batch_size, features)` |
| Input           | `(N, C, H, W)`                                                  |
| Output          | `(N, C*H*W)`                                                    |
| When to use     | Before `nn.Linear` or MLP layers after convolution/pooling      |
| What if skipped | PyTorch throws a **shape mismatch error**                       |

