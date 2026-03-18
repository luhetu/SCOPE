import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, ConcatDataset
import os
from randomaug import RandAugment


def _try_lmdb_loader(data_dir, split, transform, bs, num_workers):
    """
    If data_dir has an LMDB subdirectory for this split, use it.
    Returns DataLoader or None.
    """
    try:
        from datasets.lmdb_dataset import LMDBImageDataset
    except ImportError:
        return None
    lmdb_dir = os.path.join(data_dir + '_lmdb', split)
    meta_path = os.path.join(lmdb_dir, 'meta.pkl')
    if not os.path.isfile(meta_path):
        return None
    print(f'[LMDB] Using LMDB for {split}: {lmdb_dir}')
    ds = LMDBImageDataset(lmdb_dir, transform=transform)
    shuffle = (split == 'train')
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


def _list_class_names(root_dir):
    if not os.path.isdir(root_dir):
        return []
    return sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])


def _remap_imagefolder_targets(dataset, global_class_to_idx):
    """Remap ImageFolder targets to a shared class mapping across splits."""
    new_samples = []
    for path, old_target in dataset.samples:
        class_name = dataset.classes[old_target]
        if class_name not in global_class_to_idx:
            raise ValueError(
                f"Class '{class_name}' from split sample not found in global class mapping."
            )
        new_samples.append((path, global_class_to_idx[class_name]))

    dataset.samples = new_samples
    dataset.imgs = new_samples  # torchvision keeps imgs as alias of samples
    dataset.targets = [target for _, target in new_samples]
    dataset.class_to_idx = dict(global_class_to_idx)
    dataset.classes = [name for name, _ in sorted(global_class_to_idx.items(), key=lambda x: x[1])]

def _resolve_imagenet_splits(data_dir):
    # Prefer standard train/val if present
    standard_pairs = [
        ("train", "val"),
        ("train1", "val"),
        ("train", "val1"),
        ("train1", "val1"),
    ]
    for train_name, val_name in standard_pairs:
        train_dir = os.path.join(data_dir, train_name)
        val_dir = os.path.join(data_dir, val_name)
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            return [train_dir], val_dir

    # Support train1..train4 with last split used as val if no explicit val
    train_splits = []
    for i in range(1, 5):
        split_dir = os.path.join(data_dir, f"train{i}")
        if os.path.isdir(split_dir):
            train_splits.append(split_dir)
    if len(train_splits) >= 2:
        return train_splits[:-1], train_splits[-1]

    # Support train.X1..train.X4 with val.X
    train_splits = []
    for i in range(1, 5):
        split_dir = os.path.join(data_dir, f"train.X{i}")
        if os.path.isdir(split_dir):
            train_splits.append(split_dir)
    val_x = os.path.join(data_dir, "val.X")
    if len(train_splits) >= 1 and os.path.isdir(val_x):
        return train_splits, val_x

    expected = ", ".join([f"{a}/{b}" for a, b in standard_pairs] + ["train1..train4", "train.X1..train.X4 + val.X"])
    raise FileNotFoundError(
        f"ImageNet data_dir '{data_dir}' missing expected splits. "
        f"Tried: {expected}"
    )


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

    # Try LMDB first (faster on NFS)
    trainloader = _try_lmdb_loader(data_dir, 'train', transform_train, bs, num_workers)
    valloader   = _try_lmdb_loader(data_dir, 'val',   transform_test,  bs, num_workers)
    if trainloader is not None and valloader is not None:
        return trainloader, valloader

    # Fallback: ImageFolder
    train_dirs, val_dir = _resolve_imagenet_splits(data_dir)
    trainsets = [torchvision.datasets.ImageFolder(d, transform_train) for d in train_dirs]
    trainset = trainsets[0] if len(trainsets) == 1 else ConcatDataset(trainsets)
    valset   = torchvision.datasets.ImageFolder(val_dir, transform_test)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_workers,
                             pin_memory=True, persistent_workers=(num_workers > 0))
    valloader   = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=(num_workers > 0))
    return trainloader, valloader


def _resolve_imagenet100_splits(data_dir):
    train_dirs = []
    for i in range(1, 5):
        split_dir = os.path.join(data_dir, f"train.X{i}")
        if os.path.isdir(split_dir):
            train_dirs.append(split_dir)
    if not train_dirs:
        raise FileNotFoundError(
            f"ImageNet-100 data_dir '{data_dir}' missing train.X1..train.X4"
        )

    val_dir = os.path.join(data_dir, "val.X")
    if os.path.isdir(val_dir):
        return train_dirs, val_dir
    if len(train_dirs) >= 2:
        return train_dirs[:-1], train_dirs[-1]
    raise FileNotFoundError(
        f"ImageNet-100 data_dir '{data_dir}' missing val.X and only one train split found"
    )


def build_imagenet100_loader(data_dir, size=224, bs=256, num_workers=4, aug=False):
    # Match requested settings: RandomResizedCrop + 224x224 resize, val batch follows config
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

    train_dirs, val_dir = _resolve_imagenet100_splits(data_dir)
    trainsets = [torchvision.datasets.ImageFolder(d, transform_train) for d in train_dirs]
    trainset = trainsets[0] if len(trainsets) == 1 else ConcatDataset(trainsets)
    valset   = torchvision.datasets.ImageFolder(val_dir, transform_test)

    # Build a global class mapping across all splits, then remap each split target.
    # This supports cases where train.X1..X4 contain partial class subsets.
    class_names = set()
    for d in train_dirs + [val_dir]:
        class_names.update(_list_class_names(d))
    class_names = sorted(class_names)
    global_class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    if len(class_names) != 100:
        print(
            f"⚠️  [imagenet100] expected 100 classes, found {len(class_names)} "
            f"under {data_dir}. Continue with discovered classes."
        )

    for ts in trainsets:
        _remap_imagefolder_targets(ts, global_class_to_idx)
    _remap_imagefolder_targets(valset, global_class_to_idx)

    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_workers,
                             pin_memory=True, persistent_workers=(num_workers > 0),
                             prefetch_factor=4 if num_workers > 0 else None)
    valloader   = DataLoader(valset, batch_size=bs, shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=(num_workers > 0),
                             prefetch_factor=4 if num_workers > 0 else None)
    return trainloader, valloader


def build_cifar10_loader(data_dir, size=32, bs=128, num_workers=4, aug=False):
    """Build CIFAR-10 data loader"""
    # CIFAR-10 normalization parameters
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
    
    # Add RandAugment data augmentation
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
    """Build CIFAR-100 data loader"""
    # CIFAR-100 uses same normalization parameters as CIFAR-10
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
    
    # Add RandAugment data augmentation
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