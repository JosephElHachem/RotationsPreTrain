# alt+shift+E to run selected lines in console

import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import *
from data import *
from model import *
import os
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



def evaluate(classifier, val_loader, val_length):
    XELoss = nn.CrossEntropyLoss(reduction="mean")
    val_images, val_labels = next(iter(val_loader))
    with torch.no_grad():
        prediction = classifier(val_images)
    val_loss = XELoss(prediction, val_labels)
    predicted_labels = torch.argmax(prediction, dim=1)
    accuracy = np.round(100.0 * (predicted_labels == val_labels).sum().item() / val_length, 2)
    del val_images, val_labels
    return accuracy, val_loss


def testing(classifier, test_loader):
    XELoss = nn.CrossEntropyLoss(reduction="mean")
    test_images, test_labels = next(iter(test_loader))
    with torch.no_grad():
        prediction = classifier(test_images)
        predicted_labels = torch.argmax(prediction, dim=1)
        accuracy = 100.0 * (predicted_labels == test_labels).sum().item() / len(test_labels)
        test_loss = XELoss(prediction, test_labels)
        del test_images, test_labels, prediction
    return accuracy, test_loss


def training_fc(
        classifier,
        batch_size_l=32,
        labeled_data_ratio = 1.,
        training_data_ratio = 0.8,
        without_unlabeled = False,
        lr=1e-4,
        n_epochs=30):

    # data loaders
    (
        train_loader,
        val_loader,
        test_loader,
        train_val
    ) = data_loaders(batch_size_l, dataset='mnist', K=1,
                     batch_size_u=None, labeled_data_ratio=labeled_data_ratio,
                     training_data_ratio=training_data_ratio,
                     without_unlabeled=without_unlabeled)

    XELoss = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []

    x_counter = 0
    x_idxes = []

    x_idxes.append(x_counter)
    accuracy, val_loss = evaluate(classifier, val_loader, train_val[1])
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)

    accuracy, test_loss = testing(classifier, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

    for epoch in range(1, n_epochs+1):
        print(f"Starting epoch {epoch}")
        for local_X, local_y in iter(train_loader):
            prediction = classifier(local_X)
            loss = XELoss(prediction, local_y)

            # gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # appending losses
            train_losses.append(loss)
            x_counter += 1

        x_idxes.append(x_counter)
        accuracy, val_loss = evaluate(classifier, val_loader, train_val[1])
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        accuracy, test_loss = testing(classifier, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

    plt.figure()
    plt.title('XELosses')
    plt.xlabel('iterations')
    plt.ylabel('XE losses')
    plt.plot(train_losses, label='train')
    plt.plot(x_idxes, val_losses, label='val')
    plt.plot(x_idxes, test_losses, label='test')
    plt.legend()
    plt.savefig('training_losses.png')
    plt.show()


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

