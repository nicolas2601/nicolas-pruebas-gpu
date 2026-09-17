import torch

from pruebas.loss import collapse_std, healthy_std, negative_cosine, simsiam_loss


def test_identical_vectors_reach_the_minimum():
    p = torch.randn(8, 16)
    assert torch.isclose(negative_cosine(p, p.clone()), torch.tensor(-1.0), atol=1e-6)


def test_target_branch_receives_no_gradient():
    p = torch.randn(8, 16, requires_grad=True)
    z = torch.randn(8, 16, requires_grad=True)
    negative_cosine(p, z).backward()
    assert p.grad is not None
    assert z.grad is None


def test_loss_is_symmetric_in_the_two_views():
    p1, z1, p2, z2 = (torch.randn(8, 16) for _ in range(4))
    assert torch.isclose(simsiam_loss(p1, z1, p2, z2), simsiam_loss(p2, z2, p1, z1))


def test_collapsed_embedding_has_near_zero_std():
    z = torch.ones(64, 32)
    assert collapse_std(z).item() < 1e-6


def test_random_embedding_sits_near_the_healthy_reference():
    z = torch.randn(4096, 32)
    ratio = collapse_std(z).item() / healthy_std(32)
    assert 0.9 < ratio < 1.1
