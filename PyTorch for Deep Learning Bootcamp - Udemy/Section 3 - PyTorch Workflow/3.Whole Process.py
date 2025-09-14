

import torch
from torch import nn
import matplotlib.pyplot as plt


weight = 0.7
bias = 0.3




start = 0
end = 1
step = 0.02




X = torch.arange(start,end,step)
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
    def __int__(self):
        super().__int__()

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

    loss = loss_fn(y_pred, y_test)

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


















