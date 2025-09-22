

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
    transform = ToTensor(),
    target_transform=None
)



print(len(train_data), len(test_data))
#60000 10000



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
#torch.Size([32, 1, 28, 28]) torch.Size([32])





for batch_idx, (X, y) in enumerate(train_dataloader):
    print(f"Batch {batch_idx}:")
    print("X:", X)
    print("y:", y)
    if batch_idx == 1:
        break

"""
Batch 0:
X: tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        ...,


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]]])
y: tensor([0, 9, 5, 9, 0, 5, 5, 6, 0, 7, 8, 5, 4, 6, 3, 7, 3, 0, 2, 2, 0, 7, 2, 9,
        5, 1, 4, 5, 6, 1, 0, 9])
Batch 1:
X: tensor([[[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.4392, 0.0902, 0.0000],
          [0.0000, 0.0000, 0.0353,  ..., 0.0039, 0.1333, 0.0000],
          [0.0000, 0.0000, 0.0157,  ..., 0.0902, 0.0392, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0078, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        ...,


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0039,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0196,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]]])
y: tensor([4, 3, 0, 5, 7, 8, 1, 7, 4, 9, 2, 3, 4, 8, 8, 8, 7, 2, 4, 9, 5, 0, 4, 6,
        3, 9, 4, 3, 2, 2, 9, 0])

"""




flatten_model = nn.Flatten()

x = train_features_batch[0]
print(x.shape)
#torch.Size([1, 28, 28])


output = flatten_model(x)
print(f"shape before flattening {x.shape}")
print(f"shape after flattening {output.shape}")
"""
shape before flattening torch.Size([1, 28, 28])
shape after flattening torch.Size([1, 784])
"""





class FashMNISTModelV0(nn.Module):
    def __init__(self,input_shape:int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #We want the one imagine into a vector layer.
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )

    def forward(self,x):
        return self.layer_stack(x)




torch.manual_seed(42)

model_0 = FashMNISTModelV0(input_shape=784, #this is 28*28
                           hidden_units=10,
                           output_shape=len(class_names))



print(model_0)
"""
FashMNISTModelV0(
  (layer_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
  )
)
"""







dummy_x = torch.rand([1,1,28,28])

print(model_0(dummy_x))
"""tensor([[-0.0315,  0.3171,  0.0531, -0.2525,  0.5959,  0.2112,  0.3233,  0.2694,
         -0.1004,  0.0157]], grad_fn=<AddmmBackward0>)"""





from torchmetrics import Accuracy


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)










from timeit import default_timer as timer

def train_time(start,end):

    """Print difference between start and end time"""

    total_time = end - start

    print(f"Train time on {total_time:.3f} seconds")

    return total_time

start_time = timer()

end_time = timer()




train_time(start=start_time, end=end_time)




from tqdm.auto import tqdm

torch.manual_seed(42)

train_time_start_up = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-----")

    train_loss = 0

    for batch, (X,y) in enumerate(train_dataloader):

        model_0.train()

        y_pred = model_0(X)

        loss = loss_fn(y_pred,y)

        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0 , 0

    model_0.eval()

    with torch.inference_mode():

        for X_test,y_test in test_dataloader:

            test_pred = model_0(X_test)

            test_loss += loss_fn(test_pred, y_test)

            test_acc = Accuracy(task="multiclass", num_classes=len(train_data.classes))

            test_pred = torch.argmax(test_pred, dim=1)

            ac = test_acc(test_pred, y_test)  # tensor(1.) → 100%

            ac =+ ac


        # Calculate the test loss average per batch
        test_loss /= len(train_dataloader)

        # Calculate the test acc average per batch
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {ac:.2f}%\n")

# Calculate training time
train_time_end = timer()
total_train_time_model_0 = train_time(start=train_time_start_up,
                                            end=train_time_end)

"""
Epoch: 0
-----
  0%|          | 0/3 [00:00<?, ?it/s]Looked at 0/60000 samples.
Looked at 12800/60000 samples.
Looked at 25600/60000 samples.
Looked at 38400/60000 samples.
Looked at 51200/60000 samples.
 33%|███▎      | 1/3 [00:02<00:04,  2.22s/it]
Train loss: 0.59039 | Test loss: 0.08506, Test acc: 0.88%


Epoch: 1
-----
Looked at 0/60000 samples.
Looked at 12800/60000 samples.
Looked at 25600/60000 samples.
Looked at 38400/60000 samples.
Looked at 51200/60000 samples.
 67%|██████▋   | 2/3 [00:04<00:02,  2.32s/it]
Train loss: 0.47633 | Test loss: 0.08011, Test acc: 0.94%


Epoch: 2
-----
Looked at 0/60000 samples.
Looked at 12800/60000 samples.
Looked at 25600/60000 samples.
Looked at 38400/60000 samples.
Looked at 51200/60000 samples.

Train loss: 0.45503 | Test loss: 0.07957, Test acc: 0.94%

Train time on 7.459 seconds
100%|██████████| 3/3 [00:06<00:00,  2.25s/it]
"""






























