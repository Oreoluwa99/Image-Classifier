import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes=50, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     #16x112x112
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     #32x56x56
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     #64x28x28
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     #128x14x14
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     #256x7x7
            
            nn.Flatten(),
            
            nn.Linear(256*7*7, 3136),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(3136),
            nn.ReLU(),
            
            nn.Linear(3136, 784),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(784),
            nn.ReLU(),
            
            nn.Linear(784, num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest
#from your_module import MyModel  # Import your custom model


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders  # Assuming you have a data module
    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel()

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor"

    assert out.shape[-1] == 50, f"Expected an output tensor with last dimension 50, got {out.shape[-1]}"
