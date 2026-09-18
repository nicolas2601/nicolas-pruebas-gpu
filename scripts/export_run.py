"""Export a finished run for sharing: TensorBoard scalars as JSON, logs,
backbone-only weights, and a snapshot of the machine the run happened on.

    python scripts/export_run.py --runs ddp10 smoke --weights-from ddp10
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def shell(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def export_scalars(run_dir: Path, out: Path, run: str) -> None:
    acc = EventAccumulator(str(run_dir / "tb"))
    acc.Reload()
    scalars = {tag: [[s.step, s.value] for s in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}
    (out / f"scalars_{run}.json").write_text(json.dumps(scalars))
    for name in ("summary.json", "log.txt", "launch.log"):
        src = run_dir / name
        if src.exists():
            (out / f"{run}_{name}").write_bytes(src.read_bytes())


def export_weights(ckpt_path: Path, out: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = {k[len("backbone."):]: v for k, v in ckpt["model"].items() if k.startswith("backbone.")}
    torch.save(backbone, out / "dinov2_vits14_simsiam_imagenette_backbone.pt")
    torch.save(ckpt["model"], out / "simsiam_full_model.pt")
    return {"ckpt_epoch": ckpt["epoch"], "ckpt_step": ckpt["step"],
            "backbone_params": sum(v.numel() for v in backbone.values())}


def machine_facts() -> dict:
    return {
        "gpus": shell("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        "cuda": torch.version.cuda, "torch": torch.__version__, "python": platform.python_version(),
        "cpu": shell("lscpu | grep 'Model name' | sed 's/.*: *//'"), "cores": shell("nproc"),
        "ram": shell("free -h | awk '/Mem/{print $2}'"), "shm": shell("df -h /dev/shm | awk 'NR==2{print $2}'"),
        "os": shell(". /etc/os-release; echo $PRETTY_NAME"), "kernel": shell("uname -r"),
        "nfs": shell("df -h ~/work | awk 'NR==2{print $1, $2}'"), "hostname": platform.node(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=["ddp10", "smoke"])
    p.add_argument("--weights-from", default="ddp10")
    p.add_argument("--out", default="exports")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for run in a.runs:
        export_scalars(Path("runs") / run, out, run)
    facts = machine_facts()
    facts.update(export_weights(Path("runs") / a.weights_from / "ckpt.pt", out))
    (out / "cluster_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
