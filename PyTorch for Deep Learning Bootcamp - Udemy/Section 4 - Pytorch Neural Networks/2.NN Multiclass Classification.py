


import torch
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split



NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42



X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)


X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2,random_state=RANDOM_SEED)


# 4. Plot data
#plt.figure(figsize=(10, 7))
#plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
#plt.plot()
#plt.show()



















