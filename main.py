
import time, os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *
from RotationDataset import *
from model import *
from training_functions import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}')

# datasets
rotation_set = RotationDataset()
rotation_train, rotation_test = torch.utils.data.random_split(rotation_set, [67000, 3000])
rotation_train, rotation_val = torch.utils.data.random_split(rotation_train, [65000, 2000])

# dataloaders
rotation_dataloader = DataLoader(rotation_set, batch_size=64, shuffle=True)
rotation_trainloader = DataLoader(rotation_train, batch_size=64, shuffle=True)
rotation_valloader = DataLoader(rotation_val, batch_size=64, shuffle=False)
rotation_testloader = DataLoader(rotation_test, batch_size=64, shuffle=False)

loaders = (
    rotation_dataloader,
    rotation_trainloader,
    rotation_valloader,
    rotation_testloader
    )

# results
if not os.path.isdir('results'):
    os.mkdir('results')

dataset = 'mnist'
# model init
phi = model_phi().to(device)
classifier0 = mnist_model(phi).to(device)

# Case 1: Training only fully connected layer
freeze_model(phi)
# training
t0 = time.time()
accuracy, test_loss = training_fc(
    classifier0,
    batch_size_l=64,
    labeled_data_ratio = .1,
    training_data_ratio = 0.95,
    without_unlabeled = True,
    prefix='case1_',
    n_epochs=30)
t1 = time.time()
print(f'CASE1:: test_accuracy: {accuracy}% ; test_XELoss: {test_loss} -- time '+get_duration(t0, t1))


# Case 2: Training all network
unfreeze_model(phi)
rotation_model = rotation_model(phi).to(device)
# training
t0 = time.time()
accuracy, test_loss = training_phi(rotation_model, loaders, n_epochs=30)
t1 = time.time()
print(f'ROTATION MODEL:: test_accuracy: {accuracy}% ; test_XELoss: {test_loss} -- time '+get_duration(t0, t1))

classifier1 = mnist_model(phi).to(device) # new fc layer
accuracy, test_loss = training_fc(
    classifier1,
    batch_size_l=64,
    labeled_data_ratio = 0.1,
    training_data_ratio = 0.95,
    without_unlabeled = True,
    prefix='case2_',
    n_epochs=30)

t1 = time.time()
print(f'CASE2:: test_accuracy: {accuracy}% ; test_XELoss: {test_loss} -- total time '+get_duration(t0, t1))
