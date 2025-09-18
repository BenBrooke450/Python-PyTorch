


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







from torch import nn


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_features)
        )

    def forward(self,x):
        return self.linear_layer_stack(x)



print(X_train.shape,y_train.shape)
#torch.Size([800, 2]) torch.Size([800])


model = BlobModel(input_features=2,output_features=4,hidden_units=8)

print(model.state_dict)
"""
<bound method Module.state_dict of BlobModel(
  (linear_layer_stack): Sequential(
    (0): Linear(in_features=2, out_features=4, bias=True)
    (1): Linear(in_features=8, out_features=8, bias=True)
    (2): Linear(in_features=8, out_features=4, bias=True)
  )
)>
"""






loss_fn = nn.CrossEntropyLoss()

Optimizer = torch.optim.SGD(params = model.parameters(), lr=0.1)




model.eval()
with torch.inference_mode():
    y_logits = model(X_test)



print(y_logits[:10])
"""
tensor([[ 0.6572, -0.1953, -0.3351,  0.9652],
        [-1.1486, -1.7767, -0.8743,  0.1295],
        [-0.6236,  0.1367,  0.5047, -0.4337],
        [ 0.5848,  0.0940, -0.0602,  0.7361],
        [-0.1211, -1.3579, -0.9720,  0.8718],
        [-1.4123, -2.0735, -1.0085,  0.0440],
        [-0.3521,  0.4141,  0.6191, -0.3301],
        [-0.3814, -1.4330, -0.9211,  0.6665],
        [-1.0378, -0.2039,  0.3996, -0.6377],
        [-0.2789, -1.3819, -0.9231,  0.7355]])
"""





y_logits = model(X_test)

# Perform softmax calculation on logits across dimension 1 to get prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
"""
tensor([[ 1.6675,  0.6933, -0.4108, -0.9144],
        [ 0.5971,  0.8124, -0.2973, -0.2577],
        [-2.4009, -0.8949, -0.2591,  1.5225],
        [ 0.8217,  0.2923, -0.3890, -0.4098],
        [ 2.3012,  1.2981, -0.3854, -1.2836]], grad_fn=<SliceBackward0>)
"""


print(y_pred_probs[:5])
"""
tensor([[0.6336, 0.2392, 0.0793, 0.0479],
        [0.3253, 0.4034, 0.1330, 0.1384],
        [0.0155, 0.0698, 0.1318, 0.7829],
        [0.4590, 0.2703, 0.1368, 0.1340],
        [0.6837, 0.2507, 0.0466, 0.0190]], grad_fn=<SliceBackward0>)
"""













