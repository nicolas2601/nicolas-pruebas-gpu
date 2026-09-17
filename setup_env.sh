#!/usr/bin/env bash
# One-time environment setup on the UNABIA pod. Everything lives under the
# project directory on the NFS home, never under / (ephemeral overlay).
#
#   bash setup_env.sh
#
# Never `pip install torch` into the base conda env: it once replaced the
# cu128 build with a CPU-only one.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_DIR/.conda-env"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
mkdir -p "$TORCH_HOME" "$PROJECT_DIR/data" "$PROJECT_DIR/runs"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    conda create -y -p "$ENV_DIR" python=3.11
fi
PY="$ENV_DIR/bin/python"

"$PY" -m pip install --upgrade pip -q
"$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q -r "$PROJECT_DIR/requirements.txt"

"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a")
PY

echo "OK. Use: $PY  (or: export PATH=$ENV_DIR/bin:\$PATH)"
