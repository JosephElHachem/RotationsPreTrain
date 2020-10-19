import os
import time
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from veri_wild import DatasetFetcher
from utils import get_duration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data specs
n_channels = 3
height, width = 96, 96

# data loaders
batch_size = 128
veri_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])

# trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
#                                  download=True,
#                                  train=True,
#                                  transform=transforms.ToTensor())
trainset = DatasetFetcher(path_to_list="/data/nfs_Databases/jelhachem/veri_wild/train_test_split/train_list_start0.txt",
                          root="/data/nfs_Databases/jelhachem/veri_wild/images/train",
                          transform=veri_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

# testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
#                                  download=True,
#                                  train=False,
#                                  transform=transforms.ToTensor())
# testset = DatasetFetcher(path_to_list="/data/nfs_Databases/jelhachem/veri_wild/train_test_split/test_5000_id.txt",
#                          root="/data/nfs_Databases/jelhachem/veri_wild/images/gallery/5000_ids",
#                          transform=veri_transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

print("data loaded")
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

# rotated_targets = torch.zeros( len(trainset)+len(testset) ).to(device)
rotated_targets = torch.zeros(len(trainset)).to(device)

label_map = {
    0:0,
    90:1,
    180:2,
    270:3
}
print("starting rotations")
root = "/data/nfs_Databases/jelhachem/veri_wild/images/rotations"
images_root = os.path.join(root, "images")
labels_root = os.path.join(root, "labels")
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(images_root):
    os.mkdir(images_root)
if not os.path.isdir(labels_root):
    os.mkdir(labels_root)

idx = 0
img_idx = 0
t0 = time.time()
for local_X, _ in iter(train_loader):
    local_X.to(device)
    angle = rotations[np.random.randint(0, 4)]
    rotated_images = rotate_tensor(local_X, angle)
    rotated_targets[idx:(idx+len(local_X))] = label_map[int(angle)]
    idx += len(local_X)
    for single_image in rotated_images:
        img_path = os.path.join(images_root, str(img_idx)+".png")
        save_image(single_image, img_path)
        img_idx += 1
    print(f"saved {img_idx} images so far -- time: {get_duration(t0, time.time())}")

torch.save(rotated_targets.long(), os.path.join(labels_root, "labels.pt"))
print('done and saved')
