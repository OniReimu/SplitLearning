import torch.nn as nn
import torch.nn.functional as F
import torch

class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x = x.view(-1,1,28,28)
        x = self.pool(F.relu(self.conv1(x)))
        return x
    
class model1_sisa(nn.Module):
    def __init__(self):
        super().__init__()
        # existing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # new layer to apply max pooling
        )
        self.flatten = nn.Flatten()  # new layer to flatten the tensor

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)  # flatten the tensor here
        return x


class model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*13*13,1000)
        self.fc2 = nn.Linear(1000,100)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
    
class model2_sisa(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*13*13, 5000)
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer after fc1
        self.fc2 = nn.Linear(5000, 1000)
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer after fc2
        self.fc3 = nn.Linear(1000, 100)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # apply dropout after fc1
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # apply dropout after fc2
        x = self.fc3(x)  # additional layer

        return x


class model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc3(x)
        return x