"""SimSiam objective and the collapse indicator.

SimSiam (Chen & He, 2021): two augmented views, one encoder, a predictor on
top, negative cosine similarity with stop-gradient on the target branch.
No negatives, no momentum encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """-cos(p, stopgrad(z)), averaged over the batch. Range [-1, 1]."""
    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(
    p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor
) -> torch.Tensor:
    """Symmetrized SimSiam loss. Minimum is -1 (views agree perfectly)."""
    return 0.5 * (negative_cosine(p1, z2) + negative_cosine(p2, z1))


@torch.no_grad()
def collapse_std(z: torch.Tensor) -> torch.Tensor:
    """Mean per-dimension std of the L2-normalized embeddings.

    Healthy training sits near 1/sqrt(d). A value near 0 means every sample
    maps to the same vector: the representation collapsed.
    """
    z = F.normalize(z.float(), dim=1)
    return z.std(dim=0).mean()


def healthy_std(dim: int) -> float:
    """Reference value of collapse_std for a non-collapsed embedding."""
    return 1.0 / (dim ** 0.5)
