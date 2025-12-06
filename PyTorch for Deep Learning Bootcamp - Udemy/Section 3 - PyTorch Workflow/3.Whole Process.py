

import torch
from torch import nn
import matplotlib.pyplot as plt


weight = 0.7
bias = 0.3




start = 0
end = 1
step = 0.02




X = torch.arange(start,end,step).unsqueeze(1)
y = weight * X + bias

print(X[:10],y[:10])
"""
tensor([0.0000, 0.0200, 0.0400, 0.0600, 0.0800, 0.1000, 0.1200, 0.1400, 0.1600,
        0.1800]) tensor([0.3000, 0.3140, 0.3280, 0.3420, 0.3560, 0.3700, 0.3840, 0.3980, 0.4120,
        0.4260])
"""





train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(X_train.shape,y_train.shape,X_test.shape)
#torch.Size([40, 1]) torch.Size([40, 1]) torch.Size([10, 1]






def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14});

    plt.show()



#plot_predictions(X_train, y_train,X_test, y_test)
















class LinearRegressionModelv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1,out_features=1)

        """
        Previously we did this instead 
        
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        """
    def forward(self,x):
        return self.linear_layer(x)



torch.manual_seed(42)

model_1 = LinearRegressionModelv2()

print(model_1,model_1.state_dict())
#LinearRegressionModelv2() OrderedDict()








loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.01)

torch.manual_seed(42)

epochs = 200






for epoch in range(epochs):

    model_1.train()

    y_pred = model_1(X_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    ### TESTING

    model_1.eval()

    with torch.inference_mode():

        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred,y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

"""
Epoch: 0 | MAE Train Loss: 0.5551779866218567 | MAE Test Loss: 0.5739762187004089 
Epoch: 10 | MAE Train Loss: 0.4399680495262146 | MAE Test Loss: 0.4392663538455963 
Epoch: 20 | MAE Train Loss: 0.3247582018375397 | MAE Test Loss: 0.30455654859542847 
Epoch: 30 | MAE Train Loss: 0.20954832434654236 | MAE Test Loss: 0.16984671354293823 
Epoch: 40 | MAE Train Loss: 0.09433844685554504 | MAE Test Loss: 0.03513688966631889 
Epoch: 50 | MAE Train Loss: 0.023886388167738914 | MAE Test Loss: 0.04784907028079033 
Epoch: 60 | MAE Train Loss: 0.019956793636083603 | MAE Test Loss: 0.04580312967300415 
Epoch: 70 | MAE Train Loss: 0.016517985612154007 | MAE Test Loss: 0.037530578672885895 
Epoch: 80 | MAE Train Loss: 0.013089167885482311 | MAE Test Loss: 0.02994491532444954 
Epoch: 90 | MAE Train Loss: 0.009653175249695778 | MAE Test Loss: 0.02167237363755703 
Epoch: 100 | MAE Train Loss: 0.006215682718902826 | MAE Test Loss: 0.014086711220443249 
Epoch: 110 | MAE Train Loss: 0.002787243574857712 | MAE Test Loss: 0.005814170930534601 
Epoch: 120 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
Epoch: 130 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
Epoch: 140 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
Epoch: 150 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
Epoch: 160 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
Epoch: 170 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
Epoch: 180 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
Epoch: 190 | MAE Train Loss: 0.0012645088136196136 | MAE Test Loss: 0.013801807537674904 
"""

print(model_1.state_dict())
#OrderedDict({'linear_layer.weight': tensor([[0.6968]]), 'linear_layer.bias': tensor([0.3025])})

print(weight,bias)
#0.7 0.3











