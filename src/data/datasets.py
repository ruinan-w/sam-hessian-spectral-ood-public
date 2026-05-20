import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _dataset_spec(dataset_name):
    name = str(dataset_name).lower()
    if name == "cifar10":
        return datasets.CIFAR10, 10, CIFAR10_MEAN, CIFAR10_STD
    if name == "cifar100":
        return datasets.CIFAR100, 100, CIFAR100_MEAN, CIFAR100_STD
    raise ValueError(f"Unsupported dataset_name={dataset_name!r}; expected cifar10 or cifar100.")


def _transform(mean, std, train_augmentation):
    items = []
    if train_augmentation:
        items.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    items.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(items)


def get_classification_loaders(
    dataset_name,
    data_root,
    batch_size,
    num_workers,
    seed,
    train_augmentation=True,
):
    dataset_cls, num_classes, mean, std = _dataset_spec(dataset_name)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_dataset = dataset_cls(
        root=str(data_root),
        train=True,
        download=True,
        transform=_transform(mean, std, train_augmentation),
    )
    test_dataset = dataset_cls(
        root=str(data_root),
        train=False,
        download=True,
        transform=_transform(mean, std, False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    normalization_info = {"mean": mean, "std": std}
    return train_loader, test_loader, num_classes, normalization_info


def get_hessian_subset_loader(
    dataset_name,
    data_root,
    subset_size,
    batch_size,
    seed,
    num_workers,
):
    dataset_cls, _num_classes, mean, std = _dataset_spec(dataset_name)
    dataset = dataset_cls(
        root=str(data_root),
        train=True,
        download=False,
        transform=_transform(mean, std, False),
    )
    subset_size = min(int(subset_size), len(dataset))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(dataset), size=subset_size, replace=False)
    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
