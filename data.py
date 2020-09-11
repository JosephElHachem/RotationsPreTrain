
import math
import torch
from torchvision import datasets, transforms


def data_loaders(batch_size_l, dataset='mnist', K=1, batch_size_u=None,
                 labeled_data_ratio=1, training_data_ratio=0.8, without_unlabeled=False, unit_test=False):
    transform_train = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    no_transform = transforms.ToTensor()
    # datasets
    if dataset == 'mnist':
        all_training_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                            download=True,
                                            train=True,
                                            transform=transform_train)
    elif dataset == 'cifar':
        all_training_data = datasets.CIFAR10('~/.pytorch/CIFAR10/',
                                            download=True,
                                            train=True,
                                            transform=transform_train)
    else:
        raise ValueError('dataset wrong')

    if unit_test:
        all_training_data, _ = torch.utils.data.random_split(all_training_data, [500, len(all_training_data)-500])

    data_length = len(all_training_data)
    labeled_unlabeled = [int(data_length * labeled_data_ratio),
                         data_length - int(data_length * labeled_data_ratio)]
    labeled_data_length = labeled_unlabeled[0]
    train_val = [int(labeled_data_length * training_data_ratio),
                  labeled_data_length - int(labeled_data_length * training_data_ratio)]
    labeled_set, unlabeled_set = torch.utils.data.random_split(all_training_data, labeled_unlabeled)
    trainset, valset = torch.utils.data.random_split(labeled_set, train_val)

    if dataset == 'mnist':
        testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                         download=True,
                                         train=False,
                                         transform=no_transform)
    elif dataset == 'cifar':
        testset = datasets.CIFAR10('~/.pytorch/CIFAR10/',
                                            download=True,
                                            train=False,
                                            transform=no_transform)
    else:
        raise ValueError('dataset wrong')

    # data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_l, shuffle=True, pin_memory=True)
    if batch_size_u is None:
        batch_size_u = math.floor(labeled_unlabeled[1] / len(train_loader))
    if labeled_data_ratio < 1:
        unlabeled_loaders = [
            torch.utils.data.DataLoader(unlabeled_set, batch_size=batch_size_u, shuffle=True) for i in range(K)
            ]

    val_loader = torch.utils.data.DataLoader(valset, batch_size=len(valset), shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True, pin_memory=True)

    train_val = [len(trainset), len(valset)]

    if without_unlabeled:
        labeled_data_ratio = 1
    if labeled_data_ratio < 1:
        print(f'train labeled: {train_val[0]} -- val: {train_val[1]} -- unlabeled: {labeled_unlabeled[1]}')
        to_return = (train_loader, unlabeled_loaders, val_loader, test_loader, train_val, batch_size_u)
    else:
        print(f'train labeled: {train_val[0]} -- val: {train_val[1]}')
        to_return = (train_loader, val_loader, test_loader, train_val)
    return to_return
