#!/usr/bin/env bash
# Single-GPU smoke: 30 steps on a 4k-image subset. Validates data download,
# DINOv2 hub weights, bf16, the loss, the kNN probe and TensorBoard writing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export TORCH_HOME="$PWD/.cache/torch"
PY=".conda-env/bin/python"

CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PY" scripts/train.py \
    --data-root data --output-dir runs/smoke \
    --epochs 1 --max-steps 30 --limit-unlabeled 4096 \
    --batch-per-gpu 64 --workers 4 --knn-every 1 "$@"
