"""CLI for the SimSiam-over-DINOv2 probe run.

Single GPU smoke:
    python scripts/train.py --output-dir runs/smoke --max-steps 30 --limit-unlabeled 4096

Two GPUs:
    torchrun --standalone --nproc_per_node=2 scripts/train.py --output-dir runs/2gpu --epochs 10

Resume:
    ... --resume runs/2gpu/ckpt.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pruebas.loop import run  # noqa: E402
from pruebas.train import TrainConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="imagenette", choices=("imagenette", "stl10"))
    p.add_argument("--backbone", default="dinov2_vits14")
    p.add_argument("--no-pretrained", action="store_true", help="random init (collapse ablation)")
    p.add_argument("--resolution", type=int, default=112)
    p.add_argument("--batch-per-gpu", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-heads", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--limit-unlabeled", type=int, default=0)
    p.add_argument("--knn-every", type=int, default=2)
    p.add_argument("--resume", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = TrainConfig(
        data_root=a.data_root, output_dir=a.output_dir, dataset=a.dataset, backbone=a.backbone,
        pretrained=not a.no_pretrained, resolution=a.resolution, batch_per_gpu=a.batch_per_gpu,
        epochs=a.epochs, lr_backbone=a.lr_backbone, lr_heads=a.lr_heads,
        warmup_steps=a.warmup_steps, precision=a.precision, workers=a.workers, seed=a.seed,
        max_steps=a.max_steps, limit_unlabeled=a.limit_unlabeled, knn_every=a.knn_every,
        resume=a.resume,
    )
    summary = run(cfg)
    if summary["knn"]:  # only rank 0 evaluates; other ranks stay quiet
        print("DONE", summary["final_step"], summary["knn"][-1])


if __name__ == "__main__":
    main()
