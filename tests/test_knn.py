import pytest
import torch
import torch.nn.functional as F

from pruebas.knn import knn_top1


def _clusters(n_per_class: int, num_classes: int, dim: int, noise: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, dim, generator=gen)
    feats = centers.repeat_interleave(n_per_class, 0) + noise * torch.randn(
        n_per_class * num_classes, dim, generator=gen)
    labels = torch.arange(num_classes).repeat_interleave(n_per_class)
    return F.normalize(feats, dim=1), labels


def test_separable_clusters_are_classified_perfectly():
    tr_f, tr_y = _clusters(50, 4, 16, noise=0.05, seed=0)
    te_f, te_y = _clusters(20, 4, 16, noise=0.05, seed=0)
    assert knn_top1(tr_f, tr_y, te_f, te_y, k=20) == 1.0


def test_random_features_score_at_chance():
    gen = torch.Generator().manual_seed(1)
    tr_f = F.normalize(torch.randn(2000, 16, generator=gen), dim=1)
    tr_y = torch.randint(0, 4, (2000,), generator=gen)
    te_f = F.normalize(torch.randn(1000, 16, generator=gen), dim=1)
    te_y = torch.randint(0, 4, (1000,), generator=gen)
    assert 0.15 < knn_top1(tr_f, tr_y, te_f, te_y, k=20) < 0.35


def test_chunking_does_not_change_the_result():
    tr_f, tr_y = _clusters(50, 4, 16, noise=0.3, seed=2)
    te_f, te_y = _clusters(30, 4, 16, noise=0.3, seed=3)
    whole = knn_top1(tr_f, tr_y, te_f, te_y, k=20, chunk=10_000)
    pieces = knn_top1(tr_f, tr_y, te_f, te_y, k=20, chunk=7)
    assert whole == pieces


def test_k_larger_than_train_set_is_rejected():
    tr_f, tr_y = _clusters(2, 2, 8, noise=0.1, seed=0)
    with pytest.raises(ValueError):
        knn_top1(tr_f, tr_y, tr_f, tr_y, k=20)
