import os
import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, ConcatDataset, Sampler

from randomaug import RandAugment


class RepeatedAugSampler(Sampler):
    """
    简洁版 Repeated Augmentation Sampler。
    作用：同一个 epoch 内重复采样图像，让不同 augmentation 版本进入训练。
    注意：这里保持 epoch 长度 = len(dataset)，避免等价训练步数被放大 3 倍。
    """
    def __init__(self, dataset, num_repeats=3):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.num_samples = len(dataset)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        repeated = []
        for idx in indices:
            repeated.extend([idx] * self.num_repeats)

        repeated = torch.tensor(repeated)
        perm = torch.randperm(len(repeated), generator=g)
        repeated = repeated[perm].tolist()

        return iter(repeated[:self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _loader_extra_kwargs(num_workers):
    kwargs = {
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return kwargs


def _make_train_loader(dataset, bs, num_workers, repeated_augmentations=False):
    if repeated_augmentations:
        sampler = RepeatedAugSampler(dataset, num_repeats=3)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        **_loader_extra_kwargs(num_workers),
    )


def _make_val_loader(dataset, bs, num_workers):
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        **_loader_extra_kwargs(num_workers),
    )


def _try_lmdb_loader(
    data_dir,
    split,
    transform,
    bs,
    num_workers,
    repeated_augmentations=False,
):
    """
    If data_dir has an LMDB subdirectory for this split, use it.
    Returns DataLoader or None.
    """
    try:
        from datasets.lmdb_dataset import LMDBImageDataset
    except ImportError:
        return None

    lmdb_dir = os.path.join(data_dir + "_lmdb", split)
    meta_path = os.path.join(lmdb_dir, "meta.pkl")

    if not os.path.isfile(meta_path):
        return None

    print(f"[LMDB] Using LMDB for {split}: {lmdb_dir}")

    ds = LMDBImageDataset(lmdb_dir, transform=transform)

    if split == "train":
        return _make_train_loader(
            ds,
            bs,
            num_workers,
            repeated_augmentations=repeated_augmentations,
        )

    return _make_val_loader(ds, bs, num_workers)


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
    dataset.imgs = new_samples
    dataset.targets = [target for _, target in new_samples]
    dataset.class_to_idx = dict(global_class_to_idx)
    dataset.classes = [
        name for name, _ in sorted(global_class_to_idx.items(), key=lambda x: x[1])
    ]


def _resolve_imagenet_splits(data_dir):
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

    train_splits = []
    for i in range(1, 5):
        split_dir = os.path.join(data_dir, f"train{i}")
        if os.path.isdir(split_dir):
            train_splits.append(split_dir)

    if len(train_splits) >= 2:
        return train_splits[:-1], train_splits[-1]

    train_splits = []
    for i in range(1, 5):
        split_dir = os.path.join(data_dir, f"train.X{i}")
        if os.path.isdir(split_dir):
            train_splits.append(split_dir)

    val_x = os.path.join(data_dir, "val.X")

    if len(train_splits) >= 1 and os.path.isdir(val_x):
        return train_splits, val_x

    expected = ", ".join(
        [f"{a}/{b}" for a, b in standard_pairs]
        + ["train1..train4", "train.X1..train.X4 + val.X"]
    )

    raise FileNotFoundError(
        f"ImageNet data_dir '{data_dir}' missing expected splits. "
        f"Tried: {expected}"
    )


def _build_imagenet_transforms(size=224, aug=False, random_erasing=False):
    train_transforms = [
        transforms.RandomResizedCrop(
            size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
    ]

    if aug:
        train_transforms.append(RandAugment(2, 14))

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    if random_erasing:
        train_transforms.append(
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    transform_train = transforms.Compose(train_transforms)

    # DeiT / timm 常用验证：Resize 256 + CenterCrop 224
    eval_resize = int(size / 0.875)

    transform_test = transforms.Compose([
        transforms.Resize(eval_resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transform_train, transform_test


def build_imagenet_loader(
    data_dir,
    size=224,
    bs=256,
    num_workers=4,
    aug=False,
    random_erasing=False,
    repeated_augmentations=False,
):
    transform_train, transform_test = _build_imagenet_transforms(
        size=size,
        aug=aug,
        random_erasing=random_erasing,
    )

    # Try LMDB first
    trainloader = _try_lmdb_loader(
        data_dir,
        "train",
        transform_train,
        bs,
        num_workers,
        repeated_augmentations=repeated_augmentations,
    )

    valloader = _try_lmdb_loader(
        data_dir,
        "val",
        transform_test,
        bs,
        num_workers,
        repeated_augmentations=False,
    )

    if trainloader is not None and valloader is not None:
        return trainloader, valloader

    train_dirs, val_dir = _resolve_imagenet_splits(data_dir)

    trainsets = [
        torchvision.datasets.ImageFolder(d, transform_train)
        for d in train_dirs
    ]

    trainset = trainsets[0] if len(trainsets) == 1 else ConcatDataset(trainsets)
    valset = torchvision.datasets.ImageFolder(val_dir, transform_test)

    trainloader = _make_train_loader(
        trainset,
        bs,
        num_workers,
        repeated_augmentations=repeated_augmentations,
    )

    valloader = _make_val_loader(
        valset,
        bs,
        num_workers,
    )

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


def build_imagenet100_loader(
    data_dir,
    size=224,
    bs=256,
    num_workers=4,
    aug=False,
    random_erasing=False,
    repeated_augmentations=False,
):
    transform_train, transform_test = _build_imagenet_transforms(
        size=size,
        aug=aug,
        random_erasing=random_erasing,
    )

    train_dirs, val_dir = _resolve_imagenet100_splits(data_dir)

    trainsets = [
        torchvision.datasets.ImageFolder(d, transform_train)
        for d in train_dirs
    ]

    trainset = trainsets[0] if len(trainsets) == 1 else ConcatDataset(trainsets)
    valset = torchvision.datasets.ImageFolder(val_dir, transform_test)

    # Build a global class mapping across all splits.
    class_names = set()

    for d in train_dirs + [val_dir]:
        class_names.update(_list_class_names(d))

    class_names = sorted(class_names)
    global_class_to_idx = {
        name: idx for idx, name in enumerate(class_names)
    }

    if len(class_names) != 100:
        print(
            f"⚠️  [imagenet100] expected 100 classes, found {len(class_names)} "
            f"under {data_dir}. Continue with discovered classes."
        )

    for ts in trainsets:
        _remap_imagefolder_targets(ts, global_class_to_idx)

    _remap_imagefolder_targets(valset, global_class_to_idx)

    trainloader = _make_train_loader(
        trainset,
        bs,
        num_workers,
        repeated_augmentations=repeated_augmentations,
    )

    valloader = _make_val_loader(
        valset,
        bs,
        num_workers,
    )

    return trainloader, valloader


def build_cifar10_loader(data_dir, size=32, bs=128, num_workers=4, aug=False):
    """Build CIFAR-10 data loader."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    if aug:
        transform_train.transforms.insert(0, RandAugment(2, 14))

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    testloader = DataLoader(
        testset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return trainloader, testloader


def build_cifar100_loader(data_dir, size=32, bs=128, num_workers=4, aug=False):
    """Build CIFAR-100 data loader."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    if aug:
        transform_train.transforms.insert(0, RandAugment(2, 14))

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )

    testset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    testloader = DataLoader(
        testset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return trainloader, testloader