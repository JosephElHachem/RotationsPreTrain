import os
import torch
import os.path as osp
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from utils import read_image



class DatasetFetcher(Dataset):
    def __init__(
        self,
        path_to_images=None,
        path_to_list=None,
        root_data_from_list='/data/nfs_Databases/jelhachem/veri_wild/images/train',
        transform=transforms.ToTensor()
    ):
        '''
        Specify either path_to_images and set path_to_list=None or the opposite.
        One of the two must be None.
        '''
        super(DatasetFetcher, self).__init__()
        if path_to_images is not None and path_to_list is not None:
            raise ValueError("one of path_to_images or path_to_list must be set to None")
        self._check_before_run(path_to_list)
        self._check_before_run(path_to_images)
        if path_to_images is not None:
            self.dataset = self._get_paths_from_dir(path_to_images)
        elif path_to_list is not None:
            self.dataset = self._get_paths_from_list(path_to_list, root_data_from_list)
        else:
            raise ValueError()
        self.transform = transform
        print(f"transform used: {self.transform}")

        self.print_dataset_statistics()


    def print_dataset_statistics(self):
        length = len(self.dataset)
        print(f"------------------ dataset length: {length} ------------------")

    def _check_before_run(self, path):
        """Check if all files are available before going deeper"""
        if path is not None:
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    def _get_paths_from_dir(self, path):
        """
        a convention here is used:
        if file name contains '_', then it follows the format:
        file = label_SomeNumber.ext with ext=jpeg, jpg, png or bmp
        Otherwise, no label is considered provided and set by default to 1
        """
        print(f"------- Reading data from directory: {path} ---------")
        if path is not None:
            print('reading directory')
            dataset = []
            for file in os.listdir(path):
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    if "_" in file:
                        label = int(file.split("_")[0])
                    else:
                        label = 1
                    img_path = osp.join(path, file)
                    dataset.append((img_path, label))
            return dataset
        else:
            pass

    def _get_paths_from_list(self, path, root_data_from_list):
        print(f"------- Reading data from list {path} ---------")
        if path is not None:
            print('reading text file')
            dataset = []
            with open(path, 'r') as f:
                for line in f:
                    #format: model_id/img_path vehicle_id camera_id
                    a, vid, _ = line.split(' ')
                    img_path = osp.join(root_data_from_list, a.split('/')[1])
                    dataset.append((img_path, vid))
            return dataset
        else:
            pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, l = self.dataset[idx]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        l = int(l)
        return img, l


