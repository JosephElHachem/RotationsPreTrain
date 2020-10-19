import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import read_image

# class RotationDataset(Dataset):
#     def __init__(self):
#         self.images = torch.load('rotations/images.pt')
#         self.labels = torch.load('rotations/targets.pt')
#     def __len__(self):
#         return self.images.shape[0]
#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]

class RotationDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transform = None):
        super(RotationDataset, self).__init__()
        self.images = self._get_imgs_from_dir(imgs_path)
        self.transform = transform
        self.labels = torch.load(labels_path)

    def _get_imgs_from_dir(self, path):
        img_names = os.listdir(path)
        images = []
        for img in img_names:
            if img.endswith(".png"):
                images.append(os.path.join(path, img))
        return images

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx].item()

if __name__ == "__main__":
    imgs_path ="/data/nfs_Databases/jelhachem/veri_wild/images/rotations/images"
    labels_path = "/data/nfs_Databases/jelhachem/veri_wild/images/rotations/labels"
    data = RotationDataset(imgs_path, labels_path, transform=transforms.ToTensor())
    dataLoader = DataLoader(data, batch_size=8)
