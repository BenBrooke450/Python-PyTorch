




import sklearn
import torch

from sklearn.datasets import make_circles

n_samples = 1000

X, y = make_circles(n_samples,noise = 0.03,random_state=42)

print(len(X),len(y))
#1000 1000



print((f"First 5 samples of X: /n{X[:5]}"))
"""
First 5 samples of X: /n[[ 0.75424625  0.23148074]
 [-0.75615888  0.15325888]
 [-0.81539193  0.17328203]
 [-0.39373073  0.69288277]
 [ 0.44220765 -0.89672343]]
"""


print((f"First 5 samples of y: /n{y[:5]}"))
"""
First 5 samples of y: /n[1 1 1 1 0]
"""


print(X.shape,y.shape)
#(1000, 2) (1000,)

print(X)
"""
[[ 0.75424625  0.23148074]
 [-0.75615888  0.15325888]
 [-0.81539193  0.17328203]
 ...
 [-0.13690036 -0.81001183]
 [ 0.67036156 -0.76750154]
 [ 0.28105665  0.96382443]]
"""




X_sample = X[0]
y_sample = y[0]


print(X_sample,y_sample)
#[0.75424625 0.23148074] 1


X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)




print(len(X_train),len(X_test))
#800 200


print(n_samples)
#1000








from torch import nn

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=2,out_features=10)
        self.layer2 = nn.Linear(in_features=10,out_features=1)


    def forward(self, x):
        return self.layer2(self.layer1(x))


model_0 = CircleModelV1()

print(model_0)
"""
CircleModelV1(
  (layer1): Linear(in_features=2, out_features=10, bias=True)
  (layer2): Linear(in_features=10, out_features=1, bias=True)
)
"""








#ANOTHER WAY OF BUILDING A NN
model_1 = nn.Sequential(
    nn.Linear(in_features=2,out_features=10),
    nn.Linear(in_features=10,out_features=1)
)







#ANOTHER WAY OF BUILDING A NN

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.two_linear = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        return self.two_linear(x)


model_2 = CircleModelV1()







print(model_1.state_dict())
"""
OrderedDict({'0.weight': tensor([[-0.3263, -0.4638],
        [ 0.6815,  0.5879],
        [-0.2652, -0.1060],
        [-0.6678, -0.2900],
        [-0.1738,  0.1550],
        [-0.6500, -0.5584],
        [ 0.4334, -0.0275],
        [-0.3241,  0.4348],
        [-0.2061, -0.2671],
        [ 0.5895,  0.5377]]), 
         '0.bias': tensor([ 0.2323, -0.4856,  0.3805, -0.7037,  0.0732,  0.3536, -0.5926, -0.4121, 0.1677,  0.0897]), 
         '1.weight': tensor([[-0.2006,  0.2793, -0.0605, -0.0946, -0.2586, -0.1439, -0.0819,  0.2145, 0.0171,  0.1126]]), 
         '1.bias': tensor([0.1553])})
"""
#These are random numbers








loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc








model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)






print(y_logits[:5])
"""
tensor([[ 0.3449],
        [ 0.3429],
        [ 0.2013],
        [ 0.3862],
        [-0.0347]])
"""
#Logits






y_pred_probs = torch.sigmoid(y_logits)

#If y_pred_probs >= 0.5, y=1 (class 1)
#If y_pred_probs < 0.5, y=0 (class 0)




print(torch.round(y_pred_probs[:5]))
"""
tensor([[1.],
        [0.],
        [1.],
        [0.],
        [0.]])
"""






q = torch.round(y_pred_probs[:5]).squeeze()
print(q,"/n",y_test[:5])
#tensor([0., 0., 0., 0., 0.]) /n tensor([1., 0., 1., 0., 1.])





torch.manual_seed(42)


epoches = 100

for epoch in range(epoches):

    model_0.train()

    y_logits = model_0(X_train).squeeze()

    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,y_train) # We pass in the logits as loss_fn = nn.BCEWithLogitsLoss() wants logits and not passed through the activation function

    # loss = loss_fn(y_preds,y_train) if loss_fn = nn.BCELoss()

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    model_0.eval()

    with torch.inference_mode():

        test_logits = model_0(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

"""
Epoch: 0 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 10 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 20 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 30 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 40 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 50 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 60 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 70 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 80 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
Epoch: 90 | Loss: 0.71997, Accuracy: 50.38% | Test loss: 0.71784, Test acc: 50.00%
"""





"""import requests
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

"""


















