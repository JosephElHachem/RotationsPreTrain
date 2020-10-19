import os, time
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import *
from model import *
from RotationDataset import *
from training_functions import *

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Working on {device}')

    # datasets
    imgs_path ="/data/nfs_Databases/jelhachem/veri_wild/images/rotations/images"
    labels_path = "/data/nfs_Databases/jelhachem/veri_wild/images/rotations/labels/labels.pt"
    rotation_set = RotationDataset(imgs_path, labels_path, transform=transforms.ToTensor())
    train_size = 250000
    test_size = 15000
    val_size = len(rotation_set) - train_size - test_size

    rotation_train, rotation_test = torch.utils.data.random_split(rotation_set, [train_size+val_size, test_size])
    rotation_train, rotation_val = torch.utils.data.random_split(rotation_train, [train_size, val_size])

    # dataloaders
    rotation_trainloader = DataLoader(rotation_train, batch_size=64, shuffle=True)
    rotation_valloader = DataLoader(rotation_val, batch_size=64, shuffle=False)
    rotation_testloader = DataLoader(rotation_test, batch_size=64, shuffle=False)

    loaders = (
        rotation_trainloader,
        rotation_valloader,
        rotation_testloader
        )

    # results
    if not os.path.isdir('results'):
        os.mkdir('results')

    # model init
    phi = resnet_phi().to(device)
    rotation_resnet = rotation_model(phi).to(device)
    # training
    t0 = time.time()
    accuracy, test_loss = training_phi(rotation_resnet, loaders, n_epochs=30)
    t1 = time.time()
    print(f'ROTATION MODEL:: test_accuracy: {accuracy}% ; test_XELoss: {test_loss} -- time '+get_duration(t0, t1))
