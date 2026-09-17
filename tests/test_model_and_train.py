import pytest
import torch
from torch import nn

from pruebas.model import SimSiam, check_resolution
from pruebas.train import (
    MIN_SHM_BYTES, TrainConfig, apply_lr, build_optimizer, load_checkpoint, lr_scale,
    safe_workers, save_checkpoint, validate,
)


def test_small_shm_forces_single_process_loading():
    assert safe_workers(8, 64 * 1024 * 1024) == 0
    assert safe_workers(8, MIN_SHM_BYTES) == 8
    assert safe_workers(0, 0) == 0


class FakeBackbone(nn.Module):
    """Stands in for DINOv2: any (B, 3, H, W) -> (B, embed_dim)."""

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, x):
        return self.proj(x.mean(dim=(2, 3)))


def _model():
    return SimSiam(FakeBackbone(), proj_dim=64, proj_hidden=64, pred_hidden=16)


def test_simsiam_returns_predictions_and_projections_of_the_right_shape():
    p1, z1, p2, z2 = _model()(torch.randn(4, 3, 28, 28), torch.randn(4, 3, 28, 28))
    assert p1.shape == z1.shape == p2.shape == z2.shape == (4, 64)


def test_resolution_must_be_a_multiple_of_the_patch_size():
    check_resolution(112)
    with pytest.raises(ValueError):
        check_resolution(256)


def test_validate_rejects_bad_precision():
    with pytest.raises(ValueError):
        validate(TrainConfig(data_root="d", output_dir="o", precision="fp16"))


def test_lr_schedule_warms_up_then_decays_to_zero():
    assert lr_scale(0, 100, 10) == pytest.approx(0.1)
    assert lr_scale(9, 100, 10) == pytest.approx(1.0)
    assert lr_scale(100, 100, 10) == pytest.approx(0.0, abs=1e-9)


def test_optimizer_keeps_backbone_and_heads_in_separate_groups():
    cfg = TrainConfig(data_root="d", output_dir="o", lr_backbone=1e-5, lr_heads=1e-3)
    opt = build_optimizer(_model(), cfg)
    assert [g["lr"] for g in opt.param_groups] == [1e-5, 1e-3]
    apply_lr(opt, [1e-5, 1e-3], 0.5)
    assert [g["lr"] for g in opt.param_groups] == [5e-6, 5e-4]


def test_checkpoint_round_trip_restores_weights_and_counters(tmp_path):
    cfg = TrainConfig(data_root="d", output_dir=str(tmp_path))
    model, path = _model(), tmp_path / "ckpt.pt"
    opt = build_optimizer(model, cfg)
    save_checkpoint(path, model, opt, epoch=3, step=77, cfg=cfg)
    fresh = _model()
    epoch, step = load_checkpoint(path, fresh, build_optimizer(fresh, cfg), torch.device("cpu"))
    assert (epoch, step) == (3, 77)
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a, b)
    assert not path.with_suffix(".tmp").exists()
