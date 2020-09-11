import torch.nn as nn

class model_phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)

    def forward(self, x):
        x = nn.functional.relu((self.conv1(x)))
        x = nn.functional.relu((self.conv2(x)))
        x = nn.functional.relu((self.conv3(x)))
        x = x.view(-1, 30976)
        return x

class rotation_model(nn.Module):
    def __init__(self, Phi):
        super().__init__()
        self.conv = Phi
        self.fc = nn.Linear(30976, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class mnist_model(nn.Module):
    def __init__(self, Phi):
        super().__init__()
        self.conv = Phi
        self.fc = nn.Linear(30976, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
