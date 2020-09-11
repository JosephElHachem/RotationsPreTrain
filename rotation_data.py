import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# data loaders
batch_size = 32
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                 download=True,
                                 train=True,
                                 transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                 download=True,
                                 train=False,
                                 transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

rotations = [0, 90, 180, 270]
def rotate_tensor(tensor4D, angle):
    assert angle in [0, 90, 180, 270]
    if angle == 0:
        rotated = tensor4D
    elif angle == 90:
        rotated = tensor4D.transpose(2,3).flip(3)
    elif angle == 180:
        rotated = tensor4D.flip(2,3)
    elif angle == 270:
        rotated = tensor4D.transpose(2,3).flip(2)
    return rotated

rotated_images = torch.zeros(( len(trainset)+len(testset) , 1, 28, 28)).to(device)
rotated_targets = torch.zeros( len(trainset)+len(testset) ).to(device)

label_map = {
    0:0,
    90:1,
    180:2,
    270:3
}

idx = 0
for loader in [train_loader, test_loader]:
    for local_X, _ in iter(loader):
        angle = rotations[np.random.randint(0, 4)]
        rotated_images[idx:(idx+len(local_X))] = rotate_tensor(local_X, angle)
        rotated_targets[idx:(idx+len(local_X))] = label_map[int(angle)]
        idx += len(local_X)

if not os.path.isdir('rotations'):
    os.mkdir('rotations')
torch.save(rotated_images, 'rotations/images.pt')
torch.save(rotated_targets.long(), 'rotations/targets.pt')
print('done and saved')








