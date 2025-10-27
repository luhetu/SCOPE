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


def build_cifar10_loader(data_dir, size=32, bs=128, num_workers=4, aug=False):
    """构建CIFAR-10数据加载器"""
    # CIFAR-10的归一化参数
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 添加RandAugment数据增强
    if aug:
        transform_train.transforms.insert(0, RandAugment(2, 14))
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return trainloader, testloader


def build_cifar100_loader(data_dir, size=32, bs=128, num_workers=4, aug=False):
    """构建CIFAR-100数据加载器"""
    # CIFAR-100使用与CIFAR-10相同的归一化参数
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # 添加RandAugment数据增强
    if aug:
        transform_train.transforms.insert(0, RandAugment(2, 14))
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return trainloader, testloader