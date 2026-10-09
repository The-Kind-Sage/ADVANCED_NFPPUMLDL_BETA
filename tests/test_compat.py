import numpy as np
import torch

from utils.compat import _to_price, denorm


def test_forced_noop_passthrough():
    v = np.array([[100.0, 110.0]])
    out = _to_price(v, None, 100.0, apply=False)
    np.testing.assert_allclose(out, v)


def test_forced_inverse_group_normalizer():
    scale = torch.tensor([[5.0, 2.0]])
    out = _to_price(np.array([[0.0, 1.0]]), scale, 100.0, apply=True)
    np.testing.assert_allclose(out, [[5.0, 7.0]])


def test_auto_detect_real_space():
    out = denorm(np.array([100.0, 105.0]), None, 100.0)
    np.testing.assert_allclose(out, [100.0, 105.0])


def test_auto_detect_normalized_space():
    scale = torch.tensor([[5.0, 2.0]])
    out = denorm(np.array([[0.0, 1.0]]), scale, 100.0)
    np.testing.assert_allclose(out, [[5.0, 7.0]])
