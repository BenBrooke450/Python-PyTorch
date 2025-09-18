


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
y_blob = torch.from_numpy(y_blob).type(torch.long)


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

optimizer = torch.optim.SGD(params = model.parameters(), lr=0.1)




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


print(torch.sum(y_pred_probs[0]))
#tensor(1., grad_fn=<SumBackward0>)


print(torch.argmax(y_pred_probs[:5], 1))
#tensor([0, 1, 3, 0, 2])











def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc













torch.manual_seed(42)

epochs = 100

for epoch in range(epochs):

    model.train()

    y_logits = model(X_train)

    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)

    loss = loss_fn(y_logits,y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)

        test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits,y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
"""
Epoch: 0 | Loss: 1.82813, Acc: 26.12% | Test Loss: 0.66920, Test Acc: 87.00%
Epoch: 10 | Loss: 0.27819, Acc: 88.75% | Test Loss: 0.22941, Test Acc: 90.50%
Epoch: 20 | Loss: 0.09700, Acc: 99.12% | Test Loss: 0.08838, Test Acc: 99.50%
Epoch: 30 | Loss: 0.06836, Acc: 99.12% | Test Loss: 0.06080, Test Acc: 99.50%
Epoch: 40 | Loss: 0.05488, Acc: 99.12% | Test Loss: 0.04730, Test Acc: 99.50%
Epoch: 50 | Loss: 0.04725, Acc: 99.12% | Test Loss: 0.03947, Test Acc: 99.50%
Epoch: 60 | Loss: 0.04241, Acc: 99.12% | Test Loss: 0.03441, Test Acc: 99.50%
Epoch: 70 | Loss: 0.03910, Acc: 99.12% | Test Loss: 0.03091, Test Acc: 99.50%
Epoch: 80 | Loss: 0.03672, Acc: 99.12% | Test Loss: 0.02834, Test Acc: 99.50%
Epoch: 90 | Loss: 0.03493, Acc: 99.12% | Test Loss: 0.02638, Test Acc: 99.50%
"""






epochs = 20

for epoch in range(epochs):

    model.train()

    y_logits = model(X_train)

    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)

    if epoch % 10 == 0:
        print("Y_pred:  ",y_pred[:5])
        print("Y_softmax:  ",torch.softmax(y_logits,dim=1)[:5])

    loss = loss_fn(y_logits,y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

"""
Y_pred:   tensor([1, 0, 2, 2, 0])
Y_softmax:   tensor([[1.4380e-03, 9.9856e-01, 3.7903e-10, 5.8377e-09],
        [9.9819e-01, 2.6553e-04, 9.6836e-08, 1.5434e-03],
        [1.9149e-15, 6.3493e-19, 9.9992e-01, 8.4921e-05],
        [2.5154e-09, 3.8297e-12, 9.9109e-01, 8.9124e-03],
        [9.3376e-01, 2.6267e-04, 4.4243e-05, 6.5930e-02]],
       grad_fn=<SliceBackward0>)
Y_pred:   tensor([1, 0, 2, 2, 0])
Y_softmax:   tensor([[1.1448e-03, 9.9886e-01, 1.7968e-10, 3.0073e-09],
        [9.9855e-01, 2.0724e-04, 6.0850e-08, 1.2384e-03],
        [6.5486e-16, 1.6066e-19, 9.9993e-01, 6.6085e-05],
        [1.3402e-09, 1.6132e-12, 9.9215e-01, 7.8538e-03],
        [9.3927e-01, 2.0498e-04, 3.3810e-05, 6.0495e-02]],
       grad_fn=<SliceBackward0>)

"""









import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary



plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()
















