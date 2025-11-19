
# **`torchvision.datasets.ImageFolder` — Full Summary**

`ImageFolder` is a **dataset class in PyTorch** designed to load **images stored in a structured directory** where each subfolder represents a class. It is widely used for classification tasks when you have images organized by class labels.

---

# **1. Purpose**

`ImageFolder` simplifies:

* **Loading images** from disk
* **Automatically assigning class labels** based on folder names
* **Combining with transforms** for preprocessing and augmentation
* Creating a dataset that can be used with a **PyTorch DataLoader**

It’s commonly used for tasks like:

* Cats vs Dogs classification
* Flower dataset classification
* Any image classification dataset where images are grouped by class

---

# **2. Directory Structure Requirement**

`ImageFolder` expects a **root directory** with the following structure:

```
root/class1/xxx.png
root/class1/xxy.png
root/class1/xxz.png
root/class2/123.png
root/class2/nsdf3.png
root/class2/asd932_.png
```

* **root** → path passed to `ImageFolder`
* **class1, class2, ...** → folder names → **used as labels**
* **images inside class folders** → actual image files
* Supports nested folders, but only **first-level subfolders are considered class labels**

---

# **3. How It Works Internally**

When you instantiate:

```python
dataset = datasets.ImageFolder(root="path/to/root", transform=transform)
```

PyTorch:

1. **Scans all subfolders** of `root` to determine **classes**

   * `dataset.classes` → list of class names sorted alphabetically
   * `dataset.class_to_idx` → dictionary mapping class name → label index

2. **Collects all image file paths**

   * Supports common extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`

3. **Applies optional transforms** when retrieving an item

4. Returns **tuple `(image, label)`** when indexed:

```python
img, label = dataset[0]
```

* `img` → PIL Image (or transformed tensor if `transform` applied)
* `label` → integer corresponding to class (0, 1, 2, …)

---

# **4. Key Attributes**

| Attribute              | Description                                |
| ---------------------- | ------------------------------------------ |
| `dataset.classes`      | List of class folder names                 |
| `dataset.class_to_idx` | Dict mapping class name → label index      |
| `dataset.samples`      | List of `(image_path, class_index)` tuples |
| `dataset.targets`      | List of labels for all images              |

---

# **5. Example: Basic Usage**

```python
from torchvision import datasets, transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root="data/train", transform=transform)

# Check class names
print(dataset.classes)
# ['cats', 'dogs']

# Check mapping from class name → label
print(dataset.class_to_idx)
# {'cats': 0, 'dogs': 1}

# Access a sample
img, label = dataset[0]
print(img.shape)  # e.g., torch.Size([3,224,224])
print(label)      # 0 or 1
```

---

# **6. Using With DataLoader**

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for images, labels in train_loader:
    print(images.shape)  # [32, 3, 224, 224]
    print(labels.shape)  # [32]
    break
```

* `shuffle=True` is common for training
* `batch_size` controls batch size
* Works seamlessly with transforms

---

# **7. Combining With Augmentation**

You can apply **data augmentation transforms** when loading images:

```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(root="data/train", transform=transform)
```

* Every time you access an image, **a new random augmentation** is applied (e.g., random crop, flip)

---

# **8. Notes on Image Formats**

* `ImageFolder` relies on **PIL** internally
* Supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.ppm`, `.pgm`, `.tif`, `.tiff`, `.webp`
* Unsupported formats will be ignored (raises `OSError` if invalid)

---

# **9. Best Practices**

1. **Ensure correct directory structure**

   * Each class should have a dedicated folder
   * Folder names will define labels

2. **Apply transforms**

   * Always include `ToTensor()`
   * Include normalization for pre-trained models

3. **Use separate folders for train/val/test**

   ```
   data/train/
   data/val/
   data/test/
   ```

   * Avoid splitting manually within `ImageFolder` to prevent label mismatch

4. **Check class_to_idx**

   * Alphabetical sorting of folder names defines numeric labels
   * Example: `'cats', 'dogs'` → `[0,1]`, not necessarily the order you want

5. **Handle large datasets**

   * Combine with `DataLoader` for batching
   * Use `num_workers > 0` for parallel loading

---

# **10. Advanced Features**

* **Target Transform**

  ```python
  dataset = datasets.ImageFolder(root="data/train",
                                 transform=transform,
                                 target_transform=lambda y: y*2)
  ```

  * `target_transform` modifies label (e.g., for special encoding)

* **Custom loader**

  ```python
  datasets.ImageFolder(root="data/train",
                       loader=my_custom_loader)
  ```

  * Can replace default `PIL.Image.open` loader

---

# **11. Summary Table**

| Feature          | Description                                     |
| ---------------- | ----------------------------------------------- |
| root             | Directory containing class subfolders           |
| classes          | List of class names (alphabetical)              |
| class_to_idx     | Dict mapping class name → label index           |
| samples          | List of `(image_path, class_index)`             |
| transform        | Applied to images (PIL → Tensor / augmentation) |
| target_transform | Applied to labels                               |
| loader           | Function to load images (default PIL)           |
| **getitem**      | Returns `(image, label)` tuple                  |

---

# **12. Key Takeaways**

* `ImageFolder` is **simple and powerful** for standard image classification datasets
* Requires **organized folder structure** by class
* Supports **transforms and augmentation** via `transform` argument
* Works seamlessly with `DataLoader` for batching and shuffling
* Labels are **derived from folder names**, sorted alphabetically

