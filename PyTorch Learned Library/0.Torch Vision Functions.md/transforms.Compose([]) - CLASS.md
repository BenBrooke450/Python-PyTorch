
# **torchvision.transforms — Full Summary**

`torchvision.transforms` is a **module in PyTorch** that provides **common image transformations** for preprocessing and data augmentation. It is widely used when working with **computer vision datasets** and neural networks.

---

# **1. Purpose**

Neural networks require **tensors of fixed size and numerical type** as input. Real-world images often vary in:

* Size
* Channels (RGB, grayscale)
* Format (PIL image, NumPy array)
* Dynamic content (e.g., lighting, orientation)

`torchvision.transforms` helps with:

1. **Data Preprocessing**

   * Convert images to tensors
   * Normalize pixel values
   * Resize/crop images
2. **Data Augmentation**

   * Random rotations, flips, color jitter
   * Introduces variation to reduce overfitting
3. **Tensor Transformations**

   * Compose multiple transformations into a pipeline

---

# **2. Basic Usage**

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Explanation:**

* `Resize` → resizes the image to 224x224
* `ToTensor` → converts PIL Image or NumPy array to PyTorch tensor `[C, H, W]` with values `[0,1]`
* `Normalize` → standardizes tensor channels with mean & std

---

# **3. Transform Pipelines**

The standard approach is to use `transforms.Compose`:

```python
transform = transforms.Compose([
    transform1,
    transform2,
    transform3
])
```

* Applied **in order** from first to last
* Example pipeline:

  1. Resize
  2. RandomCrop
  3. ToTensor
  4. Normalize

---

# **4. Key Transform Classes**

## **4.1 Basic Preprocessing**

| Transform                          | Purpose                                                |
| ---------------------------------- | ------------------------------------------------------ |
| `Resize(size)`                     | Resize image to given size (tuple or int)              |
| `CenterCrop(size)`                 | Crop center of image                                   |
| `RandomCrop(size)`                 | Random crop                                            |
| `Pad(padding)`                     | Pad the image                                          |
| `Grayscale(num_output_channels=1)` | Convert to grayscale                                   |
| `ToTensor()`                       | Convert PIL/Numpy image to torch.FloatTensor `[C,H,W]` |
| `Normalize(mean, std)`             | Normalize tensor channels to zero mean/unit variance   |
| `ConvertImageDtype(dtype)`         | Convert tensor to a different type (`float`, `int`)    |

---

## **4.2 Data Augmentation / Randomization**

| Transform                                            | Purpose                                                |
| ---------------------------------------------------- | ------------------------------------------------------ |
| `RandomHorizontalFlip(p=0.5)`                        | Flip horizontally with probability p                   |
| `RandomVerticalFlip(p=0.5)`                          | Flip vertically                                        |
| `RandomRotation(degrees)`                            | Rotate randomly by ±degrees                            |
| `RandomAffine(degrees, translate, scale, shear)`     | Affine transformation: rotate, translate, scale, shear |
| `RandomResizedCrop(size, scale, ratio)`              | Crop random part and resize                            |
| `ColorJitter(brightness, contrast, saturation, hue)` | Randomly change image colors                           |
| `RandomGrayscale(p=0.1)`                             | Convert to grayscale randomly                          |

---

## **4.3 Advanced / Tensor-Specific**

| Transform                          | Purpose                                             |
| ---------------------------------- | --------------------------------------------------- |
| `RandomErasing(p, scale, ratio)`   | Randomly erases a rectangle region in tensor        |
| `GaussianBlur(kernel_size, sigma)` | Apply Gaussian blur                                 |
| `Normalize(mean, std)`             | Normalize tensor channels                           |
| `Lambda(lambd)`                    | Apply custom user-defined function                  |
| `RandomApply(transforms, p=0.5)`   | Apply a sequence of transforms with probability p   |
| `RandomChoice(transforms)`         | Apply **one** randomly chosen transform from a list |

---

# **5. Difference Between PIL and Tensor Transforms**

* **PIL Transforms:**
  Works with PIL Images (common for most torchvision datasets). Examples: `Resize`, `RandomCrop`, `RandomHorizontalFlip`

* **Tensor Transforms:**
  Works on torch tensors `[C,H,W]` after `ToTensor()`. Examples: `Normalize`, `RandomErasing`

**Important:** Always call `ToTensor()` **before** tensor-specific transforms.

---

# **6. Normalization**

Normalization rescales input tensor channels to have **zero mean and unit variance**:

```python
transforms.Normalize(mean=[0.485,0.456,0.406],
                     std=[0.229,0.224,0.225])
```

* Values are standard for pre-trained models (ImageNet)
* Input tensor must be `[0,1]` (from `ToTensor`)
* Output tensor: `(x - mean)/std`

---

# **7. Composing Complex Pipelines**

```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
```

* Random crop → horizontal flip → color jitter → tensor conversion → normalization
* Each transformation is **applied sequentially**
* Introduces stochasticity (for data augmentation)

---

# **8. Applying Transforms to Datasets**

Most `torchvision.datasets` support a `transform` argument:

```python
from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(root="./data", train=True, download=True,
                        transform=transform)
```

* Each image is automatically transformed when accessed
* Can combine with `DataLoader` for batching:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

# **9. Custom Transform Functions**

You can define your own transform using `transforms.Lambda` or a callable class:

```python
class AddNoise:
    def __call__(self, x):
        return x + 0.1*torch.randn_like(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    AddNoise()
])
```

* Custom transform can be inserted anywhere in the pipeline
* Allows complex augmentation not built-in

---

# **10. Key Points / Best Practices**

1. **Order matters:**

   * Apply PIL transforms first, then `ToTensor()`, then tensor transforms like `Normalize`

2. **Use Compose for multiple transforms**

   * Keeps code clean and reproducible

3. **Pretrained models expect normalized inputs**

   * Always normalize with the correct `mean` and `std`

4. **Random transforms introduce stochasticity**

   * Use for **training**, not usually for **validation/test**

5. **Transform consistency**

   * For paired inputs (e.g., image + mask), ensure **same random transformation** is applied to both

---

# **11. Summary Table of Common Transforms**

| Category      | Transform                            | Notes                                |
| ------------- | ------------------------------------ | ------------------------------------ |
| Preprocessing | `Resize`, `CenterCrop`, `Pad`        | Works on PIL                         |
| Conversion    | `ToTensor()`, `ConvertImageDtype`    | PIL/Numpy → Tensor                   |
| Normalization | `Normalize(mean,std)`                | Tensor values `[0,1]` → standardized |
| Augmentation  | `RandomHorizontalFlip`, `RandomCrop` | Adds stochasticity to training       |
| Color         | `ColorJitter`, `Grayscale`           | Adjust image appearance              |
| Advanced      | `RandomErasing`, `GaussianBlur`      | For tensors, adds robustness         |
| Custom        | `Lambda`, callable class             | Define user-specific transforms      |

---

# **12. Summary**

* `torchvision.transforms` = **image preprocessing + augmentation toolbox**
* Enables converting **raw images → normalized tensors** for neural networks
* Supports **randomized augmentation** for training
* Works with **PIL Images, NumPy arrays, and PyTorch tensors**
* Combined with `Compose`, it allows **flexible, reproducible pipelines**
