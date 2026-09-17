"""DINOv2 backbone plus SimSiam projector and predictor."""

from __future__ import annotations

import torch
from torch import nn

DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768}
PATCH_SIZE = 14


def check_resolution(resolution: int) -> None:
    if resolution % PATCH_SIZE != 0:
        raise ValueError(
            f"resolution {resolution} is not a multiple of {PATCH_SIZE} "
            f"(DINOv2 patch size); use 112, 154, 196 or 224"
        )


def load_dinov2(name: str = "dinov2_vits14", pretrained: bool = True) -> nn.Module:
    """Load a DINOv2 ViT from torch.hub. Forward returns the CLS embedding."""
    if name not in DINOV2_DIMS:
        raise ValueError(f"unknown backbone {name}; choose from {list(DINOV2_DIMS)}")
    backbone = torch.hub.load(DINOV2_REPO, name, pretrained=pretrained)
    backbone.embed_dim = DINOV2_DIMS[name]
    return backbone


def projector(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """3-layer SimSiam projection MLP; last BN has no affine, as in the paper."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=False),
    )


def predictor(dim: int, hidden: int) -> nn.Sequential:
    """2-layer bottleneck predictor MLP."""
    return nn.Sequential(
        nn.Linear(dim, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, dim),
    )


class SimSiam(nn.Module):
    """Any backbone exposing ``embed_dim`` and returning (B, embed_dim) features."""

    def __init__(
        self, backbone: nn.Module, proj_dim: int = 2048, proj_hidden: int = 2048,
        pred_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector(backbone.embed_dim, proj_hidden, proj_dim)
        self.predictor = predictor(proj_dim, pred_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1, z2 = self.encode(x1), self.encode(x2)
        return self.predictor(z1), z1, self.predictor(z2), z2


def build_simsiam(backbone_name: str, pretrained: bool = True) -> SimSiam:
    return SimSiam(load_dinov2(backbone_name, pretrained=pretrained))
