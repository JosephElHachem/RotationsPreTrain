import os, time
import yaml
import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import *
from model import *
from RotationDataset import *
from training_phi import training_phi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config["input_shape"] = (96, 96)
    print(config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Working on {device}')

    # datasets
    make_rotations = config["make_rotations"]
    rotations_root = config["rotations_root"]
    if make_rotations:
        original_data_transforms = transforms.Compose([transforms.Resize(config["input_shape"]),
                                                       transforms.ToTensor()])
        dataset = DatasetFetcher(path_to_images=config["dataset_path"],
                                 transform=original_data_transforms)
    else:
        dataset = None


    rotation_set = RotationDataset(dataset=dataset, transform=transforms.ToTensor(),
                                   make_rotations=make_rotations, rotations_root=rotations_root)
    print("-"*20)
    print("Rotation set is ready")

    train_size = int(len(rotation_set) * 0.85)
    val_size = int(len(rotation_set) * 0.05)
    test_size = int(len(rotation_set) * 0.1)

    rotation_train, rotation_test = torch.utils.data.random_split(rotation_set, [train_size+val_size, test_size])
    rotation_train, rotation_val = torch.utils.data.random_split(rotation_train, [train_size, val_size])

    # dataloaders
    rotation_trainloader = DataLoader(rotation_train, batch_size=config["optimizer"]["batch_size"],
                                      shuffle=True)
    rotation_valloader = DataLoader(rotation_val, batch_size=config["optimizer"]["batch_size"],
                                    shuffle=False)
    rotation_testloader = DataLoader(rotation_test, batch_size=config["optimizer"]["batch_size"],
                                     shuffle=False)

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
    accuracy, test_loss = training_phi(rotation_resnet, loaders, config)
    t1 = time.time()
    print(f'ROTATION MODEL:: test_accuracy: {accuracy}% ; test_XELoss: {test_loss} -- time '+get_duration(t0, t1))
