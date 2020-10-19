import os.path as osp
from PIL import Image, ImageFile
from torch.utils.data.dataloader import default_collate


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    else:
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
    return img

def collate_to_device(batch, device):
    return default_collate(batch).to(device)

def to_device_list(l, device):
    return [l[0].to(device), l[1].to(device)]

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def get_duration(t0, t1):
    hours, rem = divmod(t1 - t0, 3600)
    minutes, seconds = divmod(rem, 60)
    current_time = "{:0>2d}:{:0>2d}:{:0>2d}".format(int(hours), int(minutes), int(seconds))
    return current_time

