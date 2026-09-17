# nicolas_pruebas: GPU probe for the UNABIA cluster

A small, self-contained SimSiam-over-DINOv2 run whose purpose is to validate the
cluster, not to produce a thesis result: two GPUs under DDP, bf16 on Blackwell,
throughput, TensorBoard, checkpoint/resume, and a job that survives closing the
browser terminal.

## What it trains

- Backbone: DINOv2 ViT-S/14 (torch.hub, pretrained), input 112x112 (8x8 patches).
- Objective: SimSiam (negative cosine, stop-gradient, no negatives, no momentum encoder).
- Data: STL-10 unlabeled split (100k images, 96x96) for SSL; labeled train/test for a
  frozen-feature kNN probe (k=20). The pretrained backbone scores high on STL-10 before
  any training, so the signal to watch is: kNN does not drop and `collapse_ratio`
  stays near 1.0.

## Layout

```
setup_env.sh        conda env under ./.conda-env, torch cu128, TORCH_HOME under ./.cache
run_smoke.sh        1 GPU, 30 steps, 4k images
run_2gpu.sh         torchrun on 2 GPUs, detached (setsid nohup)
scripts/train.py    CLI
src/pruebas/        data, model, loss, knn, tracking, train (config/dist/ckpt), loop
tests/              CPU-only unit tests (loss, kNN, model shapes, schedule, checkpoint)
runs/<name>/        ckpt.pt, tb/, log.txt, summary.json, launch.log
```

## Run on the pod

```bash
cd ~/work/nicolas_pruebas
bash setup_env.sh
.conda-env/bin/python -m pytest
bash run_smoke.sh
bash run_2gpu.sh 2gpu --epochs 10
tail -f runs/2gpu/launch.log
.conda-env/bin/tensorboard --logdir runs --port 6006 --bind_all
```

TensorBoard through the JupyterHub proxy: `https://jupyter.unabia.unab.edu.co/user/<user>/proxy/6006/`.

## Metrics logged

| Tag | Meaning |
|---|---|
| `train/loss` | SimSiam loss, minimum -1 |
| `train/collapse_ratio` | per-dim std of normalized z over 1/sqrt(d); ~1 healthy, ~0 collapsed |
| `perf/img_per_s` | images per second across all GPUs |
| `perf/vram_gb` | peak allocated VRAM on rank 0 |
| `eval/knn_top1` | frozen-feature kNN accuracy on STL-10 test |
