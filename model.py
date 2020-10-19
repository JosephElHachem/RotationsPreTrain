import torch.nn as nn
import torchvision.models as models

class resnet_phi(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = resnet.fc.in_features
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.feature_size)
        return x

class rotation_model(nn.Module):
    def __init__(self, Phi):
        super().__init__()
        self.conv = Phi
        self.fc = nn.Linear(self.conv.feature_size, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class model_phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.feature_size = 30976

    def forward(self, x):
        x = nn.functional.relu((self.conv1(x)))
        x = nn.functional.relu((self.conv2(x)))
        x = nn.functional.relu((self.conv3(x)))
        x = x.view(-1, self.feature_size)
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
