from torch.utils.data.dataloader import default_collate

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

