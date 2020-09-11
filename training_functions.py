import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from data import *
from model import *
from utils import *
import torch

def evaluate(classifier, loader, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    XELoss = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.
    accuracy= 0.
    idx = 0.
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        idx += len(images)
        with torch.no_grad():
            prediction = classifier(images)
        loss += XELoss(prediction, labels)
        predicted_labels = torch.argmax(prediction, dim=1)
        accuracy += 100.0 * (predicted_labels == labels).sum().item()
    return np.round(accuracy/idx,2), loss.item()/idx

def training_fc(
        classifier,
        batch_size_l=32,
        labeled_data_ratio = 1.,
        training_data_ratio = 0.8,
        without_unlabeled = False,
        lr=1e-4,
        n_epochs=30,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        prefix=''):

    # data loaders
    (
        train_loader,
        val_loader,
        test_loader,
        _
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
    accuracy, val_loss = evaluate(classifier, val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)

    accuracy, test_loss = evaluate(classifier, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    t0 = time.time()
    for epoch in range(1, n_epochs+1):
        t1 = time.time()
        print(f"Starting epoch {epoch} -- "+get_duration(t0, t1))
        for local_X, local_y in iter(train_loader):
            local_X = local_X.to(device)
            local_y = local_y.to(device)

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
        accuracy, val_loss = evaluate(classifier, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        accuracy, test_loss = evaluate(classifier, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

    plt.title('XELosses')
    plt.xlabel('iterations')
    plt.ylabel('XE losses')
    plt.plot(train_losses, label='train')
    plt.plot(x_idxes, val_losses, label='val')
    plt.plot(x_idxes, test_losses, label='test')
    plt.legend()
    plt.savefig('results/'+prefix+'fc_training_losses.png')
    plt.close()

    epochs = [i+1 for i in range(len(val_accuracies))]
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('%')
    plt.plot(epochs, val_accuracies, label='val')
    plt.plot(epochs, test_accuracies, label='test')
    plt.legend()
    plt.savefig('results/'+prefix+'fc_val_accuracies.png')
    plt.close()
    return np.round(accuracy, 2), np.round(test_loss, 2)

def training_phi(
        classifier,
        loaders,
        lr=1e-4,
        n_epochs=30,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
):

    # data loaders
    (
        all_loader,
        train_loader,
        val_loader,
        test_loader
    ) = loaders

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
    accuracy, val_loss = evaluate(classifier, val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)

    accuracy, test_loss = evaluate(classifier, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    t0 = time.time()
    for epoch in range(1, n_epochs+1):
        t1 = time.time()
        print(f"Starting epoch {epoch} -- "+get_duration(t0, t1))
        for local_X, local_y in iter(train_loader):
            local_X = local_X.to(device)
            local_y = local_y.to(device)

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
        accuracy, val_loss = evaluate(classifier, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        accuracy, test_loss = evaluate(classifier, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)


    plt.title('XELosses')
    plt.xlabel('iterations')
    plt.ylabel('XE losses')
    plt.plot(train_losses, label='train')
    plt.plot(x_idxes, val_losses, label='val')
    plt.plot(x_idxes, test_losses, label='test')
    plt.legend()
    plt.savefig('results/phi_training_losses.png')
    plt.close()

    epochs = [i+1 for i in range(len(val_accuracies))]
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('%')
    plt.plot(epochs, val_accuracies, label='val')
    plt.plot(epochs, test_accuracies, label='test')
    plt.legend()
    plt.savefig('results/phi_val_accuracies.png')
    plt.close()

    return np.round(accuracy, 2), np.round(test_loss, 2)
