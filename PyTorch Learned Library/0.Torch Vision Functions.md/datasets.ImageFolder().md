**ImageFolder is doing this automatically**, and this is exactly how PyTorch is designed to work.

Let’s break down what is happening so it makes complete sense.

---

# **Why do you get labels like `tensor([0, 2, 2, 0, ...])`?**

Even though you only downloaded **PNG images**, the dataset **contains subfolders**, and **each folder name becomes a class label** automatically.

### ✔ `datasets.ImageFolder()` assigns numeric labels based on folder names:

* It **looks inside** your dataset folder
* It finds **subfolders**
* Each subfolder gets a **class index** (0, 1, 2, …)
* All images inside that subfolder automatically get the corresponding label

---

# **Let’s inspect your dataset directory**

Your path:

```
.../Dataset_BUSI_with_GT/Train_data
```

If you open this folder, you will most likely see something like:

```
Train_data/
    benign/
    malignant/
    normal/
```

PyTorch will assign:

| Folder name | Label |
| ----------- | ----- |
| `benign`    | 0     |
| `malignant` | 1     |
| `normal`    | 2     |

Order is alphabetical → so it becomes:

```
benign → 0
malignant → 1
normal → 2
```

---

# **So when you print your batch labels:**

```python
tensor([0, 2, 2, 0, 0, 0, 2, 1, 0, 0])
```

It means:

* `0` → came from **benign** folder
* `1` → came from **malignant** folder
* `2` → came from **normal** folder

You get numerical labels because **PyTorch converts folder names → class indices** automatically.

---

# **You can verify this by printing:**

```python
print(train_dataset.classes)
```

You will see a list like:

```python
['benign', 'malignant', 'normal']
```

And also:

```python
print(train_dataset.class_to_idx)
```

Example output:

```python
{'benign': 0, 'malignant': 1, 'normal': 2}
```

---



You did nothing wrong.

* The breast ultrasound dataset **contains subfolders**, each representing a class.
* `ImageFolder` automatically assigns label **indices** based on folder names.
* Those `0,1,2` labels you see come directly from the folder structure.

Everything is working perfectly.
