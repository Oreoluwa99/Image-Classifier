import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=50, dropout=0.5):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 32x112x112
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 64x56x56
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 128x28x28
        )
        
        self.fc_block = nn.Sequential(
            nn.Linear(128*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
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
