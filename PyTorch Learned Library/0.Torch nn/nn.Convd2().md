
# **What is `nn.Conv2d`?**

`nn.Conv2d` is the PyTorch layer used to scan **2D data** (images or image-like structures) using **learnable filters (kernels)**.

It is the *core* building block for CNNs.

---

# **1. Input Shape Required**

`nn.Conv2d` expects input in the following shape:

```
(batch_size, in_channels, height, width)
```

### Examples:

* RGB image 256×256 → shape: `(1, 3, 256, 256)`
* Grayscale image → `(1, 1, H, W)`
* Mini-batch of 32 RGB images → `(32, 3, H, W)`

---

# **2. The Parameters of `nn.Conv2d`**

Definition:

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1)
```

Let’s break each part down deeply:

---

## ✔ **a) `in_channels` — number of channels in input**

Examples:

* RGB image → 3 channels
* Grayscale → 1 channel
* Feature maps from a previous layer → could be 16, 32, 64, …

---

## ✔ **b) `out_channels` — number of filters**

This is how many feature detectors the layer learns.

Examples:

* 16 means the layer learns 16 filters → output has 16 channels.
* 32 means 32 filters → 32 channels output.

These filters detect different visual features:

* Edges
* Corners
* Textures
* Patterns
* Shapes

---

## ✔ **c) `kernel_size` — filter size**

Common kernel sizes:

* 3×3 (most common in modern CNNs)
* 5×5
* 7×7
* or even (height, width) tuples like `(3,5)`

### Example:

`kernel_size=3` means a 3×3 sliding window.

---

## ✔ **d) `stride` — how far the filter moves each step**

* `stride=1`: move by 1 pixel → high detail
* `stride=2`: move by 2 pixels → aggressively shrink image (downsampling)

---

## ✔ **e) `padding` — add zeros around the border**

Used to control output size.

* `padding=0` → output becomes smaller than input.
* `padding=1` with 3×3 filters keeps the output size the same.

**Why padding?**
A 3×3 filter reduces dimensions.
Padding prevents image shrinking every layer.

---

## ✔ **f) `dilation` — spacing inside the kernel**

Used rarely (for segmentation).
Expands receptive field without increasing kernel size.

---

# **3. How It Actually Calculates Output (Mechanics)**

The convolution operation does:

1. Place a kernel (example: 3×3) on top-left of the image.
2. Multiply kernel values by the overlapping pixel values.
3. Sum them → this becomes ONE output pixel.
4. Slide the kernel over the whole image (controlled by stride).
5. Repeat for each kernel → producing multiple output channels.

---

# **4. Output Shape Formula**

For each spatial dimension:

```
Out = ⌊(W - K + 2P) / S⌋ + 1
```

Where:

| Symbol | Meaning            |
| ------ | ------------------ |
| W      | input width/height |
| K      | kernel size        |
| P      | padding            |
| S      | stride             |

### Full output shape:

```
(batch_size, out_channels, out_height, out_width)
```

---

# Example Calculation

Input: `(1, 3, 32, 32)`
Layer:

```python
nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
```

Output height:

```
(32 - 3 + 2*1) / 1 + 1 = 32
```

Output shape = `(1, 16, 32, 32)`

Same size because padding=1.

---

# **5. What Conv2d Learns**

Each filter learns to detect visual patterns:

* Vertical edges
* Horizontal edges
* Diagonals
* Curves
* Blobs
* Shapes
* Textures

Deeper layers detect:

* Eyes, noses, wheels, paws
* Organs in medical scans
* Tumor edges

Eventually:

* High-level concepts (normal vs abnormal)

---

# 🎯 **6. Common Patterns in CNN Architecture**

### Typical convolution block:

```python
nn.Conv2d(...)
nn.ReLU()
nn.MaxPool2d(2)
```

### More modern:

```python
nn.Conv2d(...)
nn.BatchNorm2d(...)
nn.ReLU()
```

---

# 🧪 **7. Small Practical Example**

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # -> (B,16,32,32)
        x = F.max_pool2d(x, 2)      # -> (B,16,16,16)
        x = F.relu(self.conv2(x))   # -> (B,32,16,16)
        x = F.max_pool2d(x, 2)      # -> (B,32,8,8)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

---

# 🏁 Final Summary (Very Clear)

| Concept      | Meaning                                 |
| ------------ | --------------------------------------- |
| Conv2d       | Learns filters to scan 2D data (images) |
| in_channels  | Channels entering (RGB=3)               |
| out_channels | Number of filters learned               |
| kernel_size  | Filter size (3×3 typical)               |
| stride       | Movement step                           |
| padding      | Keeps size same or shrinks              |
| output shape | (B, out_channels, H_out, W_out)         |

