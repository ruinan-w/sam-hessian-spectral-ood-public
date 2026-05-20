from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


DEFAULT_CIFARC_ROOTS = {
    "cifar10": "data/CIFAR-10-C",
    "cifar100": "data/CIFAR-100-C",
}


class CIFARCDataset(Dataset):
    def __init__(self, dataset_name="cifar10", root=None, corruption="gaussian_noise", severity=5, transform=None):
        dataset_name = str(dataset_name).lower()
        if dataset_name not in DEFAULT_CIFARC_ROOTS:
            raise ValueError(f"Unsupported dataset_name={dataset_name!r}; expected cifar10 or cifar100.")
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}")

        self.dataset_name = dataset_name
        self.root = Path(root or DEFAULT_CIFARC_ROOTS[dataset_name])
        self.corruption = corruption
        self.severity = int(severity)
        self.transform = transform

        if not self.root.exists():
            pretty = "CIFAR-10-C" if dataset_name == "cifar10" else "CIFAR-100-C"
            raise FileNotFoundError(
                f"{pretty} directory not found: {self.root}. Download and extract it so the directory "
                "contains labels.npy and corruption .npy files."
            )

        data_path = self.root / f"{corruption}.npy"
        labels_path = self.root / "labels.npy"
        missing = [str(path) for path in (data_path, labels_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing CIFAR-C file(s) for {dataset_name}: {', '.join(missing)}. "
                f"Expected {self.root}/labels.npy and {self.root}/{corruption}.npy."
            )

        start = (self.severity - 1) * 10000
        end = self.severity * 10000
        images = np.load(data_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if len(images) < end or len(labels) < end:
            raise ValueError(
                f"{data_path} or labels.npy is too short for severity {self.severity}; "
                f"expected at least {end}, found {len(images)} images and {len(labels)} labels."
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
