

what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}


import torch

from torch import nn # nn contains all of PyTorch's building blocks for neural networks

import matplotlib.pyplot as plt


## "data (prepare and load)"


# Create *known* parameters
weight  = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)
print(X)

y = weight * X + bias

print(X[:10],y[:10])


print(len(X),len(y))
#50 50


train_split = int(0.8 * len(X))
print(train_split)
#40


X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))
#40 40 10 10


def plot_pred(train_data = X_train,
              train_labels = y_train,
              test_data = X_test,
              test_labels = y_test,
              predictions = None):

    plt.figure(figsize = (10,7))

    plt.scatter(train_data, train_labels, label="Training data")

    plt.scatter(test_data, test_labels, label ="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions)

    plt.legend(prop={"size":14})

    plt.show()


#plot_pred(X_train, y_train,X_test, y_test)







class LinearRegressionModel(nn.Module):
    def __int__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)

    def forward(self, x : torch.tensor) -> torch.tensor:
        return self.weights * x + self.bias



class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                                dtype=torch.float), # <- PyTorch loves float32 by default
                                   requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, # <- start with random bias (this will get adjusted as the model learns)
                                            dtype=torch.float), # <- PyTorch loves float32 by default
                                requires_grad=True) # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)

"""
nn.Module contains the larger building blocks (layers)
nn.Parameter contains the smaller parameters like weights and biases (put these together to make nn.Module(s))
forward() tells the larger blocks how to make calculations on inputs (tensors full of data) within nn.Module(s)
torch.optim contains optimization methods on how to improve the parameters within nn.Parameter to better represent input data
"""



#Create a random seed as we are creating these Parameters by using rand

torch.manual_seed(42)

model_0 = LinearRegressionModel()

print(list(model_0.parameters()))
"""
[Parameter containing:
tensor([0.3367], requires_grad=True), Parameter containing:
tensor([0.1288], requires_grad=True)]
"""


print(X_test)
"""
tensor([[0.8000],
        [0.8200],
        [0.8400],
        [0.8600],
        [0.8800],
        [0.9000],
        [0.9200],
        [0.9400],
        [0.9600],
        [0.9800]])
"""




with torch.inference_mode():
    y_preds = model_0(X_test)


print(y_preds)
"""
tensor([[0.3982],
        [0.4049],
        [0.4116],
        [0.4184],
        [0.4251],
        [0.4318],
        [0.4386],
        [0.4453],
        [0.4520],
        [0.4588]])
"""





#Loss function
loss_fn = nn.L1Loss()

print(loss_fn)
#L1Loss()




#Optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)







epochs = 1

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()
























