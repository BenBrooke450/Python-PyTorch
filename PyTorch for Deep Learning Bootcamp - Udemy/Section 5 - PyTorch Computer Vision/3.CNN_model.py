



import torch
from torch import nn




import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor


train_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor(),
    target_transform=None)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = torchvision.transforms.ToTensor(),
    target_transform=None
)

class_names = train_data.classes

print(class_names)
#['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']













class FashionMNISTModelV3(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.convo_block_1 = nn.Sequential(

            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.convo_block_2 = nn.Sequential(

            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)

        )


    def forward(self, x):

        x = self.convo_block_1(x)
        #print(x.shape)

        x = self.convo_block_2(x)
        #print(x.shape)

        x = self.classifier(x)
        #print(x.shape)

        return x




torch.manual_seed(42)

model_1 = FashionMNISTModelV3(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names))




image, label = train_data[0]

print(image.shape)
#torch.Size([1, 28, 28])

test_image = image

conv_layer = nn.Conv2d(in_channels=1,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)


# Print out original image shape without and with unsqueezed dimension
print(f"Test image original shape: {test_image.shape}")
#Test image original shape: torch.Size([1, 28, 28])

print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")
#Test image with unsqueezed dimension: torch.Size([1, 1, 28, 28])


max_pool_layer = nn.MaxPool2d(kernel_size=2)


test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")
#Shape after going through conv_layer(): torch.Size([1, 10, 26, 26])


test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")
#Shape after going through conv_layer() and max_pool_layer(): torch.Size([1, 10, 13, 13])

















torch.manual_seed(42)

random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"Random tensor:\n{random_tensor}")
"""
tensor([[[[0.3367, 0.1288],
          [0.2345, 0.2303]]]])
"""

print(f"Random tensor shape: {random_tensor.shape}")
#Random tensor shape: torch.Size([1, 1, 2, 2])


max_pool_layer = nn.MaxPool2d(kernel_size=2)


max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
#tensor([[[[0.3367]]]]) <- this is the maximum value from random_tensor

print(f"Max pool tensor shape: {max_pool_tensor.shape}")
#Max pool tensor shape: torch.Size([1, 1, 1, 1])









model_2 = FashionMNISTModelV3(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names))

rand_image_tensor = torch.randn(size = (1,28,28))
print(rand_image_tensor.shape)
#torch.Size([1, 28, 28])


model_2(rand_image_tensor.unsqueeze(0))
"""
torch.Size([1, 10, 14, 14])
torch.Size([1, 10, 7, 7])
torch.Size([1, 10])
"""



t = model_2(rand_image_tensor.unsqueeze(0))
print(t)
"""
tensor([[ 5.1841e-02, -3.8414e-02,  8.7882e-03, -3.8084e-03,  4.7907e-02,
          2.2627e-02, -1.7306e-02,  3.4182e-02,  4.9498e-05,  1.1735e-02]],
       grad_fn=<AddmmBackward0>)
"""









loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.1)








from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle = True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=32,
                              shuffle = False)









from torchmetrics import Accuracy

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy):

    train_loss, train_acc = 0, 0

    for batch, (X,y) in enumerate(data_loader):

        model.train()

        y_pred = model(X)

        loss = loss_fn(y_pred,y)

        train_loss += loss
        acc_setup = Accuracy(task="multiclass", num_classes=len(train_data.classes))
        max_pred = torch.argmax(y_pred, dim=1)
        train_acc += acc_setup(max_pred, y)  # tensor(1.) → 100%

        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f}")



def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy):
    test_loss, test_acc = 0, 0

    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU

            # 1. Forward pass
            test_pred_y = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred_y, y)

            acc_setup = Accuracy(task="multiclass", num_classes=len(train_data.classes))
            max_pred = torch.argmax(test_pred_y, dim=1)
            test_acc += acc_setup(max_pred, y)  # tensor(1.) → 100%

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")











epochs = 3
for epoch in range(epochs):
    print(f"{epoch}")
    train_step(model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy=Accuracy,
               )

    test_step(data_loader=test_dataloader,
              model = model_2,
              loss_fn = loss_fn,
              accuracy = Accuracy
              )

"""
0
Train loss: 1.34491 | Train acc: 0.75280
Test loss: 0.43186 | Test accuracy: 0.84%

1
Train loss: 0.75339 | Train acc: 0.86427
Test loss: 0.36121 | Test accuracy: 0.87%

2
Train loss: 0.68720 | Train acc: 0.87627
Test loss: 0.37341 | Test accuracy: 0.86%
"""







