import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from data import *
from model import *
from utils import *
import torch
from torch.utils.tensorboard import SummaryWriter

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

def training_phi(
        classifier,
        loaders,
        config,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
):

    # data loaders
    (
        train_loader,
        val_loader,
        test_loader
    ) = loaders

    writer = SummaryWriter(config["writer_path"])
    XELoss = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(classifier.parameters(), lr=config["optimizer"]["lr"],
                          momentum=config["optimizer"]["momentum"],
                          weight_decay=config["optimizer"]["weight_decay"])

    x_counter = 0

    val_accuracy, val_loss = evaluate(classifier, val_loader)
    test_accuracy, test_loss = evaluate(classifier, test_loader)

    # writer.add_scalar('loss/train', epoch_loss, 0)
    writer.add_scalar('loss/val', val_loss, 0)
    writer.add_scalar('loss/test', test_loss, 0)

    writer.add_scalar('accuracy/val', val_accuracy, 0)
    writer.add_scalar('accuracy/test', test_accuracy, 0)

    print(f"Original val accuracy: {np.round(val_accuracy, 2)}%, test loss: {np.round(val_loss, 2)}")
    print(f"Original test accuracy: {np.round(test_accuracy, 2)}%, test loss: {np.round(test_loss, 2)}")

    val_best_accuracy = 0
    test_best_accuracy = 0

    t0 = time.time()
    for epoch in range(1, config["n_epochs"]+1):
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
            x_counter += 1

            # logging
            writer.add_scalar('loss/train', loss, x_counter)

        if (epoch % config["eval_every_n_epochs"]) == 0:
            print(f"Validation and test at epoch {epoch}")
            val_accuracy, val_loss = evaluate(classifier, val_loader)
            if val_accuracy > val_best_accuracy:
                torch.save(classifier.conv.state_dict(), os.path.join(config["writer_path"], "val_model.pth"))
                val_best_accuracy  = val_accuracy

            test_accuracy, test_loss = evaluate(classifier, test_loader)
            if test_accuracy > test_best_accuracy:
                torch.save(classifier.conv.state_dict(), os.path.join(config["writer_path"], "test_model.pth"))
                test_best_accuracy = test_accuracy
                test_best_loss = test_loss

            # writer.add_scalar('loss/train', epoch_loss, 0)
            writer.add_scalar('loss/val', val_loss, x_counter)
            writer.add_scalar('loss/test', test_loss, x_counter)

            writer.add_scalar('accuracy/val', val_accuracy, x_counter)
            writer.add_scalar('accuracy/test', test_accuracy, x_counter)

            print(f"val_acc: {np.round(val_accuracy, 2)}%  val_loss: {np.round(val_loss, 2)} -- test_acc: {np.round(test_accuracy, 2)}%  test_loss: {np.round(test_loss, 2)} ---- "+get_duration(t0, t1))

    print(f"Final test accuracy: {np.round(test_accuracy, 2)}%, test loss: {np.round(test_loss, 2)}")
    return test_best_accuracy, test_best_loss





