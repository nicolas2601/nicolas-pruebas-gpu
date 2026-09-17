"""Training loop: epochs, kNN probes, checkpoint and resume."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from pruebas import data, knn
from pruebas.loss import collapse_std, healthy_std, simsiam_loss
from pruebas.model import build_simsiam
from pruebas.tracking import Tracker
from pruebas.train import (
    DistInfo, TrainConfig, apply_lr, build_optimizer, load_checkpoint, lr_scale,
    save_checkpoint, setup_distributed, teardown_distributed, unwrap, validate,
)

LOG_EVERY = 20


class Step:
    """Mutable global step shared between epochs."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


def run(cfg: TrainConfig) -> dict:
    cfg = validate(cfg)
    info = setup_distributed()
    torch.manual_seed(cfg.seed + info.rank)
    tracker = Tracker(cfg.output_dir, enabled=info.is_main)
    tracker.text(f"config {asdict(cfg)}")
    tracker.text(f"world_size={info.world_size} device={info.device}")
    try:
        return _run(cfg, info, tracker)
    finally:
        tracker.close()
        teardown_distributed(info)


def _run(cfg: TrainConfig, info: DistInfo, tracker: Tracker) -> dict:
    model = build_simsiam(cfg.backbone, cfg.pretrained).to(info.device)
    if info.distributed:
        model = DistributedDataParallel(model, device_ids=[info.local_rank])
    optimizer = build_optimizer(unwrap(model), cfg)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    pool = data.subset(data.build_unlabeled(cfg.data_root, cfg.resolution, cfg.dataset),
                       cfg.limit_unlabeled)
    loader = data.ssl_loader(pool, cfg.batch_per_gpu, cfg.workers, info.distributed, cfg.seed)
    total_steps = cfg.epochs * len(loader)
    if cfg.max_steps > 0:
        total_steps = min(total_steps, cfg.max_steps)
    tracker.text(f"pool={len(pool)} steps/epoch={len(loader)} total_steps={total_steps}")

    step, start_epoch = Step(0), 0
    ckpt_path = Path(cfg.output_dir) / "ckpt.pt"
    if cfg.resume:
        start_epoch, step.value = load_checkpoint(Path(cfg.resume), unwrap(model), optimizer, info.device)
        tracker.text(f"resumed from {cfg.resume} at epoch={start_epoch} step={step.value}")

    probe = _KnnProbe(cfg, info, tracker)
    if start_epoch == 0:
        probe.evaluate(unwrap(model).backbone, epoch=0)
    for epoch in range(start_epoch, cfg.epochs):
        if info.distributed:
            loader.sampler.set_epoch(epoch)
        _train_epoch(model, loader, optimizer, base_lrs, cfg, info, tracker, step, total_steps, epoch)
        if info.is_main:
            save_checkpoint(ckpt_path, unwrap(model), optimizer, epoch + 1, step.value, cfg)
        if (epoch + 1) % cfg.knn_every == 0 or epoch + 1 == cfg.epochs:
            probe.evaluate(unwrap(model).backbone, epoch=epoch + 1)
        if step.value >= total_steps:
            break
    summary = {"final_step": step.value, "knn": probe.history, "config": asdict(cfg)}
    tracker.dump_json("summary.json", summary)
    return summary


def _train_epoch(model, loader, optimizer, base_lrs, cfg, info, tracker, step, total_steps, epoch):
    model.train()
    use_bf16 = cfg.precision == "bf16" and info.device.type == "cuda"
    tick, seen = time.perf_counter(), 0
    for x1, x2 in loader:
        if step.value >= total_steps:
            return
        x1 = x1.to(info.device, non_blocking=True)
        x2 = x2.to(info.device, non_blocking=True)
        apply_lr(optimizer, base_lrs, lr_scale(step.value, total_steps, cfg.warmup_steps))
        with torch.autocast(info.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            p1, z1, p2, z2 = model(x1, x2)
            loss = simsiam_loss(p1.float(), z1.float(), p2.float(), z2.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step.value += 1
        seen += x1.shape[0] * info.world_size
        if step.value % LOG_EVERY == 0:
            elapsed = time.perf_counter() - tick
            _log_step(tracker, loss, z1, optimizer, seen / elapsed, info, step.value, epoch)
            tick, seen = time.perf_counter(), 0


def _log_step(tracker, loss, z1, optimizer, img_per_s, info, step, epoch):
    std = collapse_std(z1).item()
    values = {
        "train/loss": loss.item(), "train/collapse_std": std,
        "train/collapse_ratio": std / healthy_std(z1.shape[1]),
        "train/lr_backbone": optimizer.param_groups[0]["lr"],
        "train/lr_heads": optimizer.param_groups[1]["lr"],
        "perf/img_per_s": img_per_s, "train/epoch": epoch,
    }
    if info.device.type == "cuda":
        values["perf/vram_gb"] = torch.cuda.max_memory_allocated(info.device) / 1e9
    tracker.scalars(values, step)
    tracker.text(f"step={step} loss={values['train/loss']:.4f} std_ratio={values['train/collapse_ratio']:.2f} "
                 f"img/s={img_per_s:.0f} vram={values.get('perf/vram_gb', 0):.1f}GB")


class _KnnProbe:
    def __init__(self, cfg: TrainConfig, info: DistInfo, tracker: Tracker) -> None:
        self.cfg, self.info, self.tracker = cfg, info, tracker
        self.history: list[dict] = []
        self._loaders = None

    def _get_loaders(self):
        if self._loaders is None:
            train = data.build_labeled(self.cfg.data_root, self.cfg.resolution, "train", self.cfg.dataset)
            test = data.build_labeled(self.cfg.data_root, self.cfg.resolution, "test", self.cfg.dataset)
            self._loaders = (data.eval_loader(train, 256, self.cfg.workers),
                             data.eval_loader(test, 256, self.cfg.workers))
        return self._loaders

    def evaluate(self, backbone: nn.Module, epoch: int) -> None:
        if not self.info.is_main:
            self._barrier()
            return
        train_loader, test_loader = self._get_loaders()
        use_bf16 = self.cfg.precision == "bf16" and self.info.device.type == "cuda"
        tr_f, tr_y = knn.extract_features(backbone, train_loader, self.info.device, use_bf16)
        te_f, te_y = knn.extract_features(backbone, test_loader, self.info.device, use_bf16)
        acc = knn.knn_top1(tr_f, tr_y, te_f, te_y, k=self.cfg.knn_k)
        backbone.train()
        self.history.append({"epoch": epoch, "knn_top1": acc})
        self.tracker.scalar("eval/knn_top1", acc, epoch)
        self.tracker.text(f"epoch={epoch} knn_top1={acc:.4f}")
        self._barrier()

    def _barrier(self) -> None:
        if self.info.distributed:
            torch.distributed.barrier()
