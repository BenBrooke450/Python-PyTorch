

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


class LinearregressionModel(nn.Module):






















