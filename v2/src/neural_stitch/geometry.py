"""Homography helpers.

Convention used everywhere in v2: a homography H maps pixel coordinates of a source image to
pixel coordinates of a target image, ``[x', y', 1] ~ H @ [x, y, 1]``. Pixel (0, 0) is the centre of
the top-left pixel (OpenCV convention). Stored homographies use full-resolution pixels.
"""
import numpy as np
import torch


def scale_matrix(s: float) -> np.ndarray:
    """Maps pixel coordinates of an image to the same image resized by factor ``s``."""
    o = 0.5 * s - 0.5
    return np.array([[s, 0, o], [0, s, o], [0, 0, 1]], dtype=np.float64)


def normalizer(w: int, h: int) -> np.ndarray:
    """Maps pixel coordinates to roughly [-1, 1] along the long side, centred on the image."""
    s = 2.0 / max(w, h)
    return np.array([[s, 0, -s * (w - 1) / 2], [0, s, -s * (h - 1) / 2], [0, 0, 1]], dtype=np.float64)


def normalize_det(H: np.ndarray) -> np.ndarray:
    """Scale H so that det(H) = 1 (homographies are only defined up to scale)."""
    return H / np.cbrt(np.linalg.det(H))


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply H to points [N, 2]."""
    p = pts @ H[:, :2].T + H[:, 2]
    return p[:, :2] / p[:, 2:3]


def corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def _signed_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def check_warp(H: np.ndarray, w: int, h: int, max_area_change: float = 8.0) -> str | None:
    """Return a reason why H is not a plausible warp of a w x h image, or None if it looks fine."""
    c = corners(w, h)
    p = c @ H[:, :2].T + H[:, 2]
    z = p[:, 2]
    if not (np.all(z > 1e-9) or np.all(z < -1e-9)):
        return "warp sends part of the image to infinity"
    q = p[:, :2] / z[:, None]
    ratio = _signed_area(q) / _signed_area(c)
    if ratio <= 0:
        return "warp mirrors the image"
    edges = np.roll(q, -1, axis=0) - q
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return "warp folds the image"
    if ratio > max_area_change or ratio < 1 / max_area_change:
        return f"warp changes the image size too much ({ratio:.2f}x)"
    return None


# --- torch ---------------------------------------------------------------------------------------

def _sl3_basis() -> torch.Tensor:
    """8 generators of sl(3) (traceless 3x3 matrices). Same layout as v1."""
    G = torch.zeros(8, 3, 3, dtype=torch.float64)
    G[0, 0, 2] = 1  # translation x
    G[1, 1, 2] = 1  # translation y
    G[2, 0, 1] = 1  # shear
    G[3, 1, 0] = 1  # shear
    G[4, 0, 0], G[4, 1, 1] = 1, -1  # stretch
    G[5, 1, 1], G[5, 2, 2] = -1, 1  # scale
    G[6, 2, 0] = 1  # perspective x
    G[7, 2, 1] = 1  # perspective y
    return G


_SL3 = _sl3_basis()


def sl3_exp(p: torch.Tensor) -> torch.Tensor:
    """Map parameters [..., 8] to homographies [..., 3, 3] with det = 1."""
    A = torch.einsum("...k,kij->...ij", p, _SL3.to(device=p.device, dtype=p.dtype))
    return torch.linalg.matrix_exp(A)


def apply_h_t(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply homographies [..., 3, 3] to points [..., N, 2] (broadcasting over leading dims)."""
    p = pts @ H[..., :2, :2].transpose(-1, -2) + H[..., None, :2, 2]
    z = pts @ H[..., 2:3, :2].transpose(-1, -2) + H[..., None, 2:3, 2]
    return p / z
