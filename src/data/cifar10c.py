from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class CIFAR10CDataset(Dataset):
    def __init__(self, root="data/CIFAR-10-C", corruption="gaussian_noise", severity=5, transform=None):
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}")

        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C directory not found: {self.root}. "
                "Please download and extract CIFAR-10-C so this directory contains labels.npy "
                "and corruption .npy files."
            )

        data_path = self.root / f"{corruption}.npy"
        labels_path = self.root / "labels.npy"
        missing = [str(path) for path in (data_path, labels_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing CIFAR-10-C file(s): "
                + ", ".join(missing)
                + ". Please download and extract CIFAR-10-C."
            )

        start = (severity - 1) * 10000
        end = severity * 10000

        images = np.load(data_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if len(images) < end or len(labels) < end:
            raise ValueError(
                f"CIFAR-10-C files for {corruption} do not contain severity {severity}. "
                f"Expected at least {end} samples, found {len(images)} images and {len(labels)} labels."
            )

        self.images = images[start:end]
        self.labels = labels[start:end]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, label
