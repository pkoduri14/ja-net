"""Smoke test: the package imports and the model runs a forward pass (no data needed)."""
import torch
import janet


def test_public_api():
    for name in ["load_features", "JANet", "train_ja", "train_concat", "evaluate"]:
        assert hasattr(janet, name)


def test_forward_shape():
    net = janet.JANet(fdim=200, tdim=3)
    sol = torch.randn(4, 200)
    tf = torch.randn(4, 3)
    x1 = torch.rand(4)
    logS, e1, e2, C = net(sol, sol, sol, tf, x1, 1 - x1)
    assert logS.shape == (4,) and C.shape == (4, 3)
