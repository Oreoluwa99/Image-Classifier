import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes=50, dropout=0.5):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            
            nn.Linear(512, 128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        print("Shape before linear layers:", x.shape)
        x = self.fc_layers(x)
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
