

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






print(train_data.classes)
#['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class_names = train_data.classes




image, label = train_data[0]
print(image.shape,label)
#torch.Size([1, 28, 28]) 9





print(train_data,test_data)
"""
Dataset FashionMNIST
    Number of datapoints: 60000
    Root location: data
    Split: Train
    StandardTransform
Transform: ToTensor() Dataset FashionMNIST
    Number of datapoints: 10000
    Root location: data
    Split: Test
    StandardTransform
Transform: ToTensor()
"""













from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle = True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=32,
                              shuffle = False)









# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {32}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {32}")
"""
Dataloaders: (<torch.utils.data.dataloader.DataLoader object at 0x1313d0950>, <torch.utils.data.dataloader.DataLoader object at 0x104ab4d70>)
Length of train dataloader: 1875 batches of 32
Length of test dataloader: 313 batches of 32
"""


train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape,train_labels_batch.shape)







class FashionMNISTV(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape : int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU())

    def forward(self,x:torch.Tensor):
        return self.layer_stack(x)




torch.manual_seed(42)

model_1 = FashionMNISTV(input_shape=784,
                         hidden_units=10,
                         output_shape=len(class_names))





from torchmetrics import Accuracy

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.1)



from timeit import default_timer as timer

def train_time(start,end):

    """Print difference between start and end time"""

    total_time = end - start

    print(f"Train time on {total_time:.3f} seconds")

    return total_time





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



torch.manual_seed(42)

from tqdm.auto import tqdm




train_time_start = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader,
               model=model_1,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy=Accuracy
               )


    test_step(data_loader=test_dataloader,
              model = model_1,
              loss_fn = loss_fn,
              accuracy = Accuracy
              )

train_time_end = timer()

train_time(start=train_time_start,end=train_time_end)








