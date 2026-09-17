"""STL-10 loaders: two-view SSL pairs from the unlabeled split, plain
tensors from the labeled train/test splits for the kNN probe."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def two_view_transform(resolution: int) -> T.Compose:
    """SimSiam augmentation recipe (crop, flip, jitter, gray, blur)."""
    kernel = max(3, (resolution // 10) | 1)  # odd kernel, ~10% of the side
    return T.Compose([
        T.RandomResizedCrop(resolution, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(resolution: int) -> T.Compose:
    return T.Compose([
        T.Resize(resolution),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TwoViews(Dataset):
    """Wraps an image dataset; returns (view1, view2) of the same image."""

    def __init__(self, base: Dataset, transform: T.Compose) -> None:
        self.base, self.transform = base, transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, _ = self.base[index]
        return self.transform(image), self.transform(image)


def build_unlabeled(root: str | Path, resolution: int, download: bool = True) -> TwoViews:
    base = datasets.STL10(str(root), split="unlabeled", download=download)
    return TwoViews(base, two_view_transform(resolution))


def build_labeled(root: str | Path, resolution: int, split: str, download: bool = True):
    return datasets.STL10(str(root), split=split, download=download,
                          transform=eval_transform(resolution))


def ssl_loader(
    dataset: Dataset, batch_size: int, workers: int, distributed: bool, seed: int
) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed) if distributed else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=workers, pin_memory=True, drop_last=True, persistent_workers=workers > 0,
    )


def eval_loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def subset(dataset: Dataset, limit: int, seed: int = 0) -> Dataset:
    """Deterministic random subset; used by the smoke run."""
    if limit <= 0 or limit >= len(dataset):
        return dataset
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=gen)[:limit].tolist()
    return torch.utils.data.Subset(dataset, idx)
