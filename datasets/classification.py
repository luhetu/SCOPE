import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import os
from randomaug import RandAugment

def build_imagenet_loader(data_dir, size=224, bs=256, num_workers=4, aug=False):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    if aug:
        transform_train.transforms.insert(1, RandAugment(2, 14))

    trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform_train)
    valset   = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform_test)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader   = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, valloader
