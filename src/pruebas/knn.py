"""Frozen-feature kNN evaluation (weighted voting, as in DINO)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module, loader, device: torch.device, use_bf16: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalized features and labels for every sample in the loader."""
    backbone.eval()
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = backbone(images)
        feats.append(F.normalize(out.float(), dim=1).cpu())
        labels.append(targets.clone())
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_top1(
    train_f: torch.Tensor, train_y: torch.Tensor,
    test_f: torch.Tensor, test_y: torch.Tensor,
    k: int = 20, temperature: float = 0.07, chunk: int = 1024,
) -> float:
    """Top-1 accuracy of a cosine kNN with exp(sim/T) vote weights."""
    if train_f.shape[0] < k:
        raise ValueError(f"k={k} larger than the {train_f.shape[0]} train samples")
    num_classes = int(max(train_y.max(), test_y.max()).item()) + 1
    correct = 0
    for start in range(0, test_f.shape[0], chunk):
        sims = test_f[start:start + chunk] @ train_f.T
        top_sim, top_idx = sims.topk(k, dim=1)
        votes = _weighted_votes(top_sim, train_y[top_idx], num_classes, temperature)
        correct += (votes.argmax(dim=1) == test_y[start:start + chunk]).sum().item()
    return correct / test_f.shape[0]


def _weighted_votes(
    top_sim: torch.Tensor, top_labels: torch.Tensor, num_classes: int, temperature: float
) -> torch.Tensor:
    weights = (top_sim / temperature).exp()
    one_hot = F.one_hot(top_labels, num_classes).float()
    return (one_hot * weights.unsqueeze(-1)).sum(dim=1)
