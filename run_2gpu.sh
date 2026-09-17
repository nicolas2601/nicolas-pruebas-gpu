#!/usr/bin/env bash
# Two-GPU DDP run, detached from the terminal so closing the browser tab does
# not kill it. Logs go to runs/<name>/launch.log; TensorBoard to runs/<name>/tb.
#
#   bash run_2gpu.sh 2gpu --epochs 10
#   tail -f runs/2gpu/launch.log
#   .conda-env/bin/tensorboard --logdir runs --port 6006 --bind_all
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
NAME="${1:?usage: run_2gpu.sh <run-name> [train args]}"; shift
export TORCH_HOME="$PWD/.cache/torch"
mkdir -p "runs/$NAME"

setsid nohup .conda-env/bin/torchrun --standalone --nproc_per_node=2 scripts/train.py \
    --data-root data --output-dir "runs/$NAME" "$@" \
    > "runs/$NAME/launch.log" 2>&1 &
echo "launched pid $! -> runs/$NAME/launch.log"
