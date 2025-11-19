
# **FULL, DETAILED SUMMARY OF `torchinfo.summary`**

`torchinfo` (formerly “torch-summary”) is the **modern, actively maintained replacement** for the older `torchsummary` library.
It produces a **layer-by-layer, hierarchical, detailed summary** of a PyTorch model.

It is the **closest PyTorch equivalent** to Keras’s:

```python
model.summary()
```

But **much more powerful**.

---

# **1. How You Use It**

Basic usage:

```python
from torchinfo import summary

summary(model, input_size=(32, 3, 224, 224))
```

This prints:

* Layer type
* Output shape
* Input shape
* Number of parameters
* Trainable / non-trainable params
* MACs / FLOPs (optional)
* Layer depth / hierarchy
* Forward pass memory

---

# **2. What Makes `torchinfo` Better Than `torchsummary`**

`torchinfo`:

✔ Handles **multiple inputs**
✔ Handles **multiple outputs**
✔ Handles **RNNs, LSTMs, GRUs**
✔ Handles **Transformers, attention blocks, dynamic graphs**
✔ More robust shape detection
✔ Offers many more columns
✔ Can show **kernel size**, **padding**, **stride**, etc.
✔ Has **depth-based hierarchical tree view**
✔ Active project, still maintained

---

# **3. How `torchinfo.summary` Works Internally**

This is the key difference:
`torchinfo` performs a **structured recursive model traversal** combined with **hook-based shape capture**, but with a much more advanced design compared to `torchsummary`.

### **3.1. It recursively walks the entire model tree**

It uses:

```python
model.named_modules()
```

This discovers **every submodule**, not just `.children()`.

### **3.2. It attaches forward hooks only where needed**

It avoids double-counting by recognizing:

* reused modules
* shared layers
* container layers (Sequential, ModuleList)

### **3.3. It runs a simulated forward pass**

It creates a dummy input (or uses your actual input) and runs:

```
output = model(input)
```

Hooks extract:

* Input shape
* Output shape
* Parameter counts
* MACs/FLOPs (if enabled)

### **3.4. It merges the hook results with the recursive model structure**

This creates a hierarchical summary, respecting:

* module nesting
* branch structures
* skip connections
* sequential blocks

---

# **4. What `torchinfo.summary()` Reports**

The summary is **customizable**, but common columns include:

### **4.1. Layer information**

* Layer name
* Layer type (Conv2d, Linear, GRU, MultiheadAttention, etc.)
* Depth (indentation shows hierarchy)
* Module index

### **4.2. Shape information**

* `input_size`
* `output_size`
* Number of parameters
* Groups, kernel size, stride, padding (for Conv)

### **4.3. Parameter breakdown**

* Trainable
* Non-trainable

### **4.4. Computational cost**

(If enabled)

* Multiply-accumulate operations (MACs)
* FLOPs
* Total per-layer compute cost

### **4.5. Memory**

* Estimated activation memory
* Weight memory
* Gradient memory

---

# **5. Important Arguments (Critical)**

### **input_size**

Required for shape inference if your model doesn’t infer internally.

```python
summary(model, input_size=(32, 3, 224, 224))
```

### **depth**

Controls how deep the tree goes:

```python
summary(model, input_size=(32,3,224,224), depth=4)
```

### **col_names**

You can show/hide information columns:

```python
summary(model,
        input_size=(1,3,224,224),
        col_names=["input_size", "output_size", "num_params", "kernel_size"])
```

### **verbose**

Controls verbosity:

* 0 → only totals
* 1 → top-level and immediate children
* 2 → full hierarchical

### **dtypes**

Allows you to specify FP16, FP32, etc.

---

# **6. What `torchinfo` Can Handle That `torchsummary` Cannot**

### **6.1 Multiple inputs**

Example:

```python
summary(model, 
        input_data=[torch.randn(1,3,224,224), torch.randn(1,10)])
```

### **6.2 Multiple outputs**

```python
return out1, out2
```

Torchsummary breaks; torchinfo handles it.

---

### **6.3 RNNs, LSTMs, GRUs**

`torchsummary` notoriously breaks here.
`torchinfo` handles hidden states and time dimensions.

---

### **6.4 Transformers**

Multi-head attention layers, positional encodings, and token sequences work fine.

---

### **6.5 Dynamic models**

Conditional flows and loops also work better, though not perfect.

---

# **7. Typical Output (Explained)**

Example (simplified):

```
==========================================================================================
Layer (type)                      Input Shape           Output Shape        Param #  
==========================================================================================
├─Conv2d: 1-1                     [32, 3, 224, 224]     [32, 64, 112, 112]  1,792
├─BatchNorm2d: 1-2                [32, 64, 112, 112]    [32, 64, 112, 112]  128
├─ReLU: 1-3                       [32, 64, 112, 112]    [32, 64, 112, 112]  0
├─MaxPool2d: 1-4                  [32, 64, 112, 112]    [32, 64, 56, 56]    0
│
├─Sequential: 1-5
│  ├─Conv2d: 2-1                  [32, 64, 56, 56]      [32, 128, 56, 56]   73,856
│  ├─ReLU: 2-2                    [32, 128, 56, 56]     [32, 128, 56, 56]   0
│  └─Conv2d: 2-3                  [32, 128, 56, 56]     [32, 128, 56, 56]   147,584
│
├─Linear: 1-6                     [32, 50176]           [32, 1000]          50,177,000
==========================================================================================
Total params: 50,400,360
Trainable params: 50,400,360
Non-trainable params: 0
==========================================================================================
```

This shows:

* Hierarchy (`├─`, `│`, etc.)
* Input and output shapes
* Parameters per layer
* Total parameter counts

---

# **8. Installation**

```
pip install torchinfo
```

---

# **9. When You Should Use `torchinfo.summary()`**

You should use it when:

✔ You want to debug layer shapes
✔ You want to detect mismatches in tensor dimensions
✔ You want a clean model summary for a research paper
✔ You have a CNN, RNN, LSTM, GRU, Transformer, UNet, etc.
✔ You want to monitor computational cost (MACs/FLOPs)
✔ You want a Keras-like summary in PyTorch

---

# **10. When `torchinfo` Might Struggle**

Although vastly better than `torchsummary`, it may still struggle with:

* Models where shapes depend on runtime data (rare)
* Models using very dynamic control flow
* Models that change shape depending on internal conditions

These are limitations of PyTorch’s tracing and hook systems in general.

