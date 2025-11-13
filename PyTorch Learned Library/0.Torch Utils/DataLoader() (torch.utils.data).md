
# **PyTorch `DataLoader` Summary**

## **1️⃣ What is DataLoader?**

In PyTorch, `DataLoader` is a **utility class** that provides an **iterable over a dataset**, allowing you to easily **batch, shuffle, and load data in parallel**.

It is typically used with `Dataset` objects (like `torchvision.datasets` or custom datasets) to feed data into models during **training** or **evaluation**.

---

## **2️⃣ Main Purpose**

* Efficiently **load data in batches**.
* **Shuffle** the data to improve model training.
* Support **parallel data loading** via multiple worker processes.
* Optionally apply **collate functions** to handle custom batching logic.

---

## **3️⃣ Key Parameters**

| Parameter     | Description                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------- |
| `dataset`     | The dataset to load from (must be a subclass of `torch.utils.data.Dataset`).                |
| `batch_size`  | Number of samples per batch (default `1`).                                                  |
| `shuffle`     | Whether to shuffle the data at every epoch (`True` or `False`).                             |
| `num_workers` | Number of subprocesses to use for data loading (`0` means load in main process).            |
| `drop_last`   | If `True`, drops the last incomplete batch if dataset size isn’t divisible by `batch_size`. |
| `collate_fn`  | Function to merge a list of samples into a batch (useful for variable-length inputs).       |
| `pin_memory`  | If `True`, moves tensors to CUDA pinned memory for faster GPU transfer.                     |
| `sampler`     | Optional: specify the strategy to sample indices (instead of `shuffle`).                    |

---

## **4️⃣ How DataLoader Works**

1. **Indexing the Dataset**

   * `DataLoader` internally uses the dataset’s `__getitem__` method to retrieve data samples.

2. **Batching**

   * Data is grouped into batches of size `batch_size`.

3. **Shuffling**

   * If `shuffle=True`, the order of indices is randomized each epoch.

4. **Parallel Loading**

   * `num_workers>0` allows loading batches in **parallel** using Python multiprocessing.

5. **Collating**

   * The `collate_fn` merges individual samples into a batch. Default is `default_collate`, which stacks tensors along the first dimension.

---

## **5️⃣ Example: Using DataLoader**

Let’s create a simple dataset and load it with `DataLoader`:

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 1️⃣ Create a custom dataset
class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(10).float()  # Data: 0,1,...,9
        self.labels = torch.arange(10).float() * 2  # Labels: 0,2,...,18

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2️⃣ Instantiate the dataset
dataset = MyDataset()

# 3️⃣ Create a DataLoader
loader = DataLoader(
    dataset,
    batch_size=3,      # 3 samples per batch
    shuffle=True,      # shuffle data every epoch
    num_workers=0      # load data in main process
)

# 4️⃣ Iterate through batches
for batch_idx, (X, y) in enumerate(loader):
    print(f"Batch {batch_idx}:")
    print("X:", X)
    print("y:", y)
```

**Sample Output** (shuffled):

```
Batch 0:
X: tensor([3., 0., 7.])
y: tensor([ 6.,  0., 14.])
Batch 1:
X: tensor([2., 8., 1.])
y: tensor([ 4., 16.,  2.])
...
```

---

## **6️⃣ Tips & Best Practices**

* **Batch Size:** Larger batch sizes speed up training but use more GPU memory.
* **Shuffle:** Always shuffle **training data**, but not necessary for test/validation data.
* **num\_workers:** On Mac/Windows, start small (`0`–`4`) to avoid multiprocessing issues; Linux can handle more.
* **Pin Memory:** Use `pin_memory=True` when transferring batches to GPU:

  ```python
  for X, y in loader:
      X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
  ```
* **Custom Collate:** Useful for handling variable-length sequences (like NLP or graphs).

---

**Summary**

* `DataLoader` = batching + shuffling + parallel loading + optional collate.
* Works seamlessly with custom datasets.
* Crucial for **efficient training of neural networks** in PyTorch.

