import torch
from torch.utils.data import Dataset

class RotationDataset(Dataset):
    def __init__(self):
        self.images = torch.load('rotations/images.pt')
        self.labels = torch.load('rotations/targets.pt')
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
