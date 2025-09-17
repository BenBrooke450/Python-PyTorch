




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




class CircleModelv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10)

        self.layer_2 = nn.Linear(in_features=10,out_features=10)

        self.layer_3 = nn.Linear(in_features=10,out_features=1)


    def forward(self,x):
        # z = self.layer_1(x)
        # z = self.layer_2(x)
        # z = self.layer_3(x)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x)))


model_3 = CircleModelv2()

print(model_3)
"""
CircleModelv2(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=10, bias=True)
  (layer_3): Linear(in_features=10, out_features=1, bias=True)
)
"""

print(model_3.state_dict())
"""
OrderedDict({'layer_1.weight': tensor([[ 0.5406,  0.5869],
        [-0.1657,  0.6496],
        [-0.1549,  0.1427],
        [-0.3443,  0.4153],
        [ 0.6233, -0.5188],
        [ 0.6146,  0.1323],
        [ 0.5224,  0.0958],
        [ 0.3410, -0.0998],
        [ 0.5451,  0.1045],
        [-0.3301,  0.1802]]), 'layer_1.bias': tensor([-0.3258, -0.0829, -0.2872,  0.4691, -0.5582, -0.3260, -0.1997, -0.4252,
         0.0667, -0.6984]), 'layer_2.weight': tensor([[ 0.2856, -0.2686,  0.2441,  0.0526, -0.1027,  0.1954,  0.0493,  0.2555,
          0.0346, -0.0997],
        [ 0.0850, -0.0858,  0.1331,  0.2823,  0.1828, -0.1382,  0.1825,  0.0566,
          0.1606, -0.1927],
        [-0.3130, -0.1222, -0.2426,  0.2595,  0.0911,  0.1310,  0.1000, -0.0055,
          0.2475, -0.2247],
        [ 0.0199, -0.2158,  0.0975, -0.1089,  0.0969, -0.0659,  0.2623, -0.1874,
         -0.1886, -0.1886],
        [ 0.2844,  0.1054,  0.3043, -0.2610, -0.3137, -0.2474, -0.2127,  0.1281,
          0.1132,  0.2628],
        [-0.1633, -0.2156,  0.1678, -0.1278,  0.1919, -0.0750,  0.1809, -0.2457,
         -0.1596,  0.0964],
        [ 0.0669, -0.0806,  0.1885,  0.2150, -0.2293, -0.1688,  0.2896, -0.1067,
         -0.1121, -0.3060],
        [-0.1811,  0.0790, -0.0417, -0.2295,  0.0074, -0.2160, -0.2683, -0.1741,
         -0.2768, -0.2014],
        [ 0.3161,  0.0597,  0.0974, -0.2949, -0.2077, -0.1053,  0.0494, -0.2783,
         -0.1363, -0.1893],
        [ 0.0009, -0.1177, -0.0219, -0.2143, -0.2171, -0.1845, -0.1082, -0.2496,
          0.2651, -0.0628]]), 'layer_2.bias': tensor([ 0.2721,  0.0985, -0.2678,  0.2188, -0.0870, -0.1212, -0.2625, -0.3144,
         0.0905, -0.0691]), 'layer_3.weight': tensor([[ 0.1231, -0.2595,  0.2348, -0.2321, -0.0546,  0.0661,  0.1633,  0.2553,
          0.2881, -0.2507]]), 'layer_3.bias': tensor([0.0796])})
"""


loss_fn_2 = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_3.parameters(),lr = 0.1)


torch.manual_seed(42)

epochs = 1000


for epoch in range(epochs):

    model_3.train()

    y_logits = model_3(X_train).squeeze()

    y_pred = torch.round((torch.sigmoid(y_logits)))

    loss = loss_fn_2(y_logits,y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_1.eval()
    with torch.inference_mode():

        test_logits = model_1(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

"""
Epoch: 0 | Loss: 0.69396, Accuracy: 50.88% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 100 | Loss: 0.69305, Accuracy: 50.38% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 200 | Loss: 0.69299, Accuracy: 51.12% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 300 | Loss: 0.69298, Accuracy: 51.62% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 400 | Loss: 0.69298, Accuracy: 51.12% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 500 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 600 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 700 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 800 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.70071, Test acc: 46.50%
Epoch: 900 | Loss: 0.69298, Accuracy: 51.00% | Test loss: 0.70071, Test acc: 46.50%
"""












###################################### TESTING LINEAR NN



weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # linear regression formula

# Check the data
print(len(X_regression))
X_regression[:5], y_regression[:5]


# Create train and test splits
train_split = int(0.8 * len(X_regression)) # 80% of data used for training set
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Check the lengths of each split
print(len(X_train_regression),
    len(y_train_regression),
    len(X_test_regression),
    len(y_test_regression))


# Same architecture as model_1 (but using nn.Sequential)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
)

# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)

# Train the model
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put data to target device
X_train_regression, y_train_regression = X_train_regression, y_train_regression
X_test_regression, y_test_regression = X_test_regression, y_test_regression

for epoch in range(epochs):
    ### Training
    # 1. Forward pass
    y_pred = model_2(X_train_regression)

    # 2. Calculate loss (no accuracy since it's a regression problem, not classification)
    loss = loss_fn(y_pred, y_train_regression)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_2.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model_2(X_test_regression)
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test_regression)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")

"""
Epoch: 0 | Train loss: 0.75986, Test loss: 0.54143
Epoch: 100 | Train loss: 0.09309, Test loss: 0.02901
Epoch: 200 | Train loss: 0.07376, Test loss: 0.02850
Epoch: 300 | Train loss: 0.06745, Test loss: 0.00615
Epoch: 400 | Train loss: 0.06107, Test loss: 0.02004
Epoch: 500 | Train loss: 0.05698, Test loss: 0.01061
Epoch: 600 | Train loss: 0.04857, Test loss: 0.01326
Epoch: 700 | Train loss: 0.06109, Test loss: 0.02127
Epoch: 800 | Train loss: 0.05599, Test loss: 0.01426
Epoch: 900 | Train loss: 0.05571, Test loss: 0.00603
"""

######################################





















