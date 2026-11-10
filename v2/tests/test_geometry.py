import numpy as np
import torch

from neural_stitch.geometry import apply_h, apply_h_t, check_warp, normalize_det, scale_matrix, sl3_exp


def test_sl3_exp_identity_and_unit_det():
    assert torch.allclose(sl3_exp(torch.zeros(8, dtype=torch.float64)), torch.eye(3, dtype=torch.float64))
    H = sl3_exp(torch.randn(5, 8, dtype=torch.float64) * 0.3)
    assert torch.allclose(torch.linalg.det(H), torch.ones(5, dtype=torch.float64))


def test_apply_numpy_and_torch_agree():
    H = np.array([[1.1, 0.1, 5], [-0.05, 0.9, -3], [1e-4, 2e-4, 1]])
    pts = np.random.default_rng(0).random((10, 2)) * 300
    a = apply_h(H, pts)
    b = apply_h_t(torch.tensor(H), torch.tensor(pts)).numpy()
    assert np.allclose(a, b)


def test_scale_matrix_matches_pixel_centres():
    # Pixel 0 of a 2x-downscaled image covers full-res pixels 0 and 1, so its centre is at 0.5.
    S = scale_matrix(0.5)
    assert np.allclose(apply_h(S, np.array([[0.5, 0.5]])), [[0.0, 0.0]])
    assert np.allclose(apply_h(np.linalg.inv(S), np.array([[0.0, 0.0]])), [[0.5, 0.5]])


def test_normalize_det():
    H = np.diag([2.0, 2.0, 2.0])
    assert np.isclose(np.linalg.det(normalize_det(H)), 1.0)


def test_check_warp():
    assert check_warp(np.eye(3), 100, 80) is None
    assert "mirror" in check_warp(np.diag([-1.0, 1, 1]), 100, 80)
    assert "size" in check_warp(np.diag([10.0, 10, 1]), 100, 80)
    assert "infinity" in check_warp(np.array([[1, 0, 0], [0, 1, 0], [-0.02, 0, 1.0]]), 100, 80)
