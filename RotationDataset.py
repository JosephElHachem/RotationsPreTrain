import os, time
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import get_duration
from utils import read_image
from veri_wild import DatasetFetcher


class CustomLoader(Dataset):
    def __init__(self):
        super(CustomLoader, self).__init__()
        self.images = torch.load('rotations/images.pt')
        self.labels = torch.load('rotations/targets.pt')
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class RotationDataset(Dataset):
    """
    Pytorch Dataset class that generates and encapsules the rotated data
    in a pytorch dataset class

    Should be followed by a dataloader
    """
    def __init__(self, dataset=None, transform=transforms.ToTensor(),
                 make_rotations=False, rotations_root=None):
        """
        *dataset: pytorch dataset object to construct rotations from
        *transform: transform to be applied on dataloader of rotations
        *make_rotations: flag, set to True if the class should create the rotations
                        sef to False if rotations are already created
        *rotations_root: str, root to load (and save rotations)
        """
        super(RotationDataset, self).__init__()
        if rotations_root is None and not make_rotations:
            raise InputError("cant have rotations_root not set and make_rotations=False")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        self.root = rotations_root
        self.images_root = os.path.join(self.root, "images")
        self.labels_root = os.path.join(self.root, "labels")

        if make_rotations:
            self.datalength = len(dataset)
            self.generate_and_save_rotation()


        self.images = self._get_imgs_from_dir(self.images_root)
        self.transform = transform
        self.labels = torch.load(os.path.join(self.labels_root, "labels.pt"))

    def _get_imgs_from_dir(self, path):
        print(f"Loading images from {path}")
        img_names = os.listdir(path)
        images = []
        labels = []
        if "_" in img_names[0]:
            label_present_in_name = True
        else:
            label_present_in_name = False

        for img in img_names:
            if img.endswith(".png"):
                img_name = img.split(".")[0]
                if label_present_in_name:
                    img_name, label = img_name.split("_")
                else:
                    label = "1"
                images.append(int(img_name))
                labels.append(int(label))

        images, labels = (list(t) for t in zip(*sorted(zip(images, labels))))
        if label_present_in_name:
            sorted_images = [os.path.join(path, str(img)+"_"+str(label)+".png") for img, label in zip(images, labels)]
        else:
            sorted_images = [os.path.join(path, str(img)+".png") for img in images]
        return sorted_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx].item()

    def rotate_tensor(self, tensor4D, angle):
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

    def generate_and_save_rotation(self):
        rotated_targets = torch.zeros(self.datalength).to(self.device)

        rotations = [0, 90, 180, 270]
        label_map = {
            0:0,
            90:1,
            180:2,
            270:3
        }

        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        if not os.path.isdir(self.images_root):
            os.mkdir(self.images_root)
        if not os.path.isdir(self.labels_root):
            os.mkdir(self.labels_root)

        t0 = time.time()
        idx = 0
        img_idx = 0
        t0 = time.time()
        batch_idx = 0
        for local_X, _ in iter(self.dataloader):
            local_X.to(self.device)
            angle = rotations[np.random.randint(0, 4)]
            rotated_images = self.rotate_tensor(local_X, angle)
            rotated_targets[idx:(idx+len(local_X))] = label_map[int(angle)]
            idx += len(local_X)
            for single_image in rotated_images:
                img_path = os.path.join(self.images_root, str(img_idx)+ "_" + str(angle) + ".png")
                save_image(single_image, img_path)
                img_idx += 1
            print(f"Batch {batch_idx}/{len(self.dataloader)} -- saved {img_idx} images so far -- time: {get_duration(t0, time.time())}")
            batch_idx += 1

        torch.save(rotated_targets.long(), os.path.join(self.labels_root, "labels.pt"))
        print('done and saved')


if __name__=="__main__":
    height, width = 96, 96
    veri_transform = transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor()])
    dataset = None
    make_rotations = False
    rotations_root ="/data/nfs_Databases/jelhachem/veri_wild/images/rotations"
    data = RotationDataset(dataset=dataset,
                           make_rotations=make_rotations,
                           rotations_root=rotations_root,
                           transform=veri_transform)
    dataLoader = DataLoader(data, batch_size=8)
