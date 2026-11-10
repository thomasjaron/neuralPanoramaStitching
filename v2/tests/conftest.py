"""Synthetic test data: overlapping views of one textured image with known homographies."""
import cv2
import numpy as np
import pytest

from neural_stitch.config import StitchConfig
from neural_stitch.geometry import apply_h, corners
from neural_stitch.images import InputImage


def texture(seed: int, w: int, h: int) -> np.ndarray:
    """Random multi-scale noise plus shapes: enough corners for SIFT."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for cell in (64, 16, 4):
        noise = rng.random((h // cell + 2, w // cell + 2, 3)).astype(np.float32)
        img += cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC) / 3
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    for _ in range(120):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), int(rng.integers(4, 25)), color, -1)
        else:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 40)), y + int(rng.integers(5, 40))), color, -1)
    return img


def view_transform(tx: float, ty: float, angle: float = 0.0, scale: float = 1.0,
                   px: float = 0.0, py: float = 0.0) -> np.ndarray:
    """G maps view pixels -> big-image pixels."""
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty], [px, py, 1.0]])


def make_view(big: np.ndarray, G: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.warpPerspective(big, np.linalg.inv(G), (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


def corner_error(H_est: np.ndarray, H_true: np.ndarray, w: int, h: int) -> float:
    c = corners(w, h)
    return float(np.linalg.norm(apply_h(H_est, c) - apply_h(H_true, c), axis=1).mean())


@pytest.fixture(scope="session")
def scene():
    """Three overlapping 320x240 views (given in scrambled order) plus one unrelated image."""
    big = texture(1, 900, 520)
    Gs = [view_transform(300, 140, 0.02, 1.0, 1e-5, 0),
          view_transform(80, 120, -0.03, 1.05),
          view_transform(500, 160, 0.05, 0.95, 0, 2e-5)]
    views = [make_view(big, G, 320, 240) for G in Gs]
    images = [InputImage("b_middle", views[0]), InputImage("unrelated", texture(99, 320, 240)),
              InputImage("a_left", views[1]), InputImage("c_right", views[2])]
    truth = {0: Gs[0], 2: Gs[1], 3: Gs[2]}  # image index -> view-to-big
    return images, truth


@pytest.fixture
def fast_config() -> StitchConfig:
    cfg = StitchConfig(device="cpu")
    cfg.match.backend = "sift"
    cfg.match.min_inliers = 30
    n = cfg.neural
    n.work_side, n.hidden, n.layers, n.freqs = 96, 32, 2, 4
    n.patches_per_image, n.c2f_iters, n.warp_warmup = 16, 20, 10
    n.min_iters, n.max_iters, n.check_every, n.preview_every, n.preview_side = 20, 60, 20, 1, 64
    cfg.output.max_side = 400
    return cfg
