



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
            nn.Linear(in_features=hidden_units*0,
                      out_features=output_shape)

        )


    def forward(self, x):

        x = self.convo_block_1(x)
        print(x)

        x = self.convo_block_2(x)
        print(x)

        x = self.classider(x)

        return x




torch.manual_seed(42)

model_1 = FashionMNISTModelV3(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names))





