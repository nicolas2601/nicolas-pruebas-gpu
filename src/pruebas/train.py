"""Config, distributed setup, optimizer and checkpoints for the SimSiam run."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from pruebas.model import check_resolution


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    output_dir: str
    backbone: str = "dinov2_vits14"
    pretrained: bool = True
    resolution: int = 112
    batch_per_gpu: int = 128
    epochs: int = 10
    lr_backbone: float = 1e-5
    lr_heads: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    precision: str = "bf16"
    workers: int = 8
    seed: int = 42
    max_steps: int = 0          # >0 caps the total steps (smoke)
    limit_unlabeled: int = 0    # >0 subsamples the SSL pool (smoke)
    knn_every: int = 2          # epochs between kNN probes
    knn_k: int = 20
    resume: str = ""


def validate(cfg: TrainConfig) -> TrainConfig:
    check_resolution(cfg.resolution)
    if cfg.precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be bf16 or fp32, got {cfg.precision}")
    if cfg.batch_per_gpu <= 0 or cfg.epochs <= 0:
        raise ValueError("batch_per_gpu and epochs must be positive")
    return cfg


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistInfo:
    """torchrun sets RANK/WORLD_SIZE; without them we run a single process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        return DistInfo(rank, world_size, local_rank, torch.device("cpu"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return DistInfo(rank, world_size, local_rank, torch.device("cuda", local_rank))


def teardown_distributed(info: DistInfo) -> None:
    if info.distributed and dist.is_initialized():
        dist.destroy_process_group()


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Small LR on the pretrained backbone, larger on the fresh heads."""
    groups = [
        {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
        {"params": list(model.projector.parameters()) + list(model.predictor.parameters()),
         "lr": cfg.lr_heads},
    ]
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then cosine decay to zero; multiplies each group's base lr."""
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def apply_lr(optimizer: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    epoch: int, step: int, cfg: TrainConfig,
) -> None:
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "step": step, "config": asdict(cfg),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a killed run never leaves a half-written file


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> tuple[int, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload["epoch"], payload["step"]


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model
