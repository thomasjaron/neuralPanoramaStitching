"""Turn homographies into a panorama image. The canvas size comes from the warped image corners."""
import math
from collections.abc import Callable

import cv2
import numpy as np
import torch

from .config import RESOLUTIONS, OutputConfig
from .geometry import apply_h, corners, scale_matrix
from .images import resize_max


def canvas_bounds(Hs: list[np.ndarray], sizes: list[tuple[int, int]]) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of all warped images, in reference pixels."""
    pts = np.concatenate([apply_h(H, corners(w, h)) for H, (w, h) in zip(Hs, sizes, strict=True)])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return float(x0), float(y0), float(x1), float(y1)


def native_size(bounds: tuple[float, float, float, float]) -> tuple[int, int]:
    """Panorama size (w, h) at the reference image's own pixel scale."""
    x0, y0, x1, y1 = bounds
    return round(x1 - x0 + 1), round(y1 - y0 + 1)


def output_size(native: tuple[int, int], cfg: OutputConfig) -> tuple[tuple[int, int], str | None]:
    """Output size (w, h) for the chosen resolution, and a note if it had to be limited."""
    W0, H0 = native
    note = None
    exact = None
    if cfg.resolution == "native":
        s = min(1.0, cfg.max_side / max(W0, H0))
    else:
        target = RESOLUTIONS[cfg.resolution][0]
        s = target / min(W0, H0)
        exact = target
        if s > 1:  # never upscale: there is no real detail to add
            s, exact = 1.0, None
            note = f"{cfg.resolution} is larger than the images allow, so the native size was used"
    if W0 * H0 * s * s > cfg.max_megapixels * 1e6:
        s = math.sqrt(cfg.max_megapixels * 1e6 / (W0 * H0))
        exact = None
        note = f"limited to {cfg.max_megapixels:g} megapixels"
    w, h = max(1, round(W0 * s)), max(1, round(H0 * s))
    if exact:  # make the short side exactly the preset despite rounding
        if W0 >= H0:
            h = exact
        else:
            w = exact
    return (w, h), note


def output_transform(bounds: tuple[float, float, float, float],
                     cfg: OutputConfig) -> tuple[np.ndarray, tuple[int, int], str | None]:
    """Matrix T mapping reference pixels to output pixels, the output size (w, h) and a note."""
    x0, y0, _, _ = bounds
    W0, H0 = native_size(bounds)
    (w, h), note = output_size((W0, H0), cfg)
    sx, sy = w / W0, h / H0
    T = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])
    return T, (w, h), note


def render_classic(images: list[np.ndarray], Hs: list[np.ndarray], T: np.ndarray,
                   size: tuple[int, int]) -> np.ndarray:
    """Warp every image onto the canvas and feather-blend (weights fall off towards image edges).

    Returns RGBA uint8; alpha marks pixels covered by at least one image.
    """
    w, h = size
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    s_out = T[0, 0]
    for img, H in zip(images, Hs, strict=True):
        # Shrink first so warping does not alias when the output is smaller than the input.
        small, s = resize_max(img, max(1, math.ceil(max(img.shape[:2]) * min(1.0, s_out * 1.5))))
        M = T @ H @ np.linalg.inv(scale_matrix(s))
        mask = np.zeros((small.shape[0] + 2, small.shape[1] + 2), np.uint8)
        mask[1:-1, 1:-1] = 1
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)[1:-1, 1:-1].astype(np.float32)
        weight = cv2.warpPerspective(dist, M, (w, h), flags=cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(small, M, (w, h), flags=cv2.INTER_LINEAR)
        acc += warped.astype(np.float32) * weight[..., None]
        wsum += weight
    rgb = np.clip(acc / np.maximum(wsum, 1e-6)[..., None], 0, 255).astype(np.uint8)
    alpha = np.where(wsum > 0, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


@torch.no_grad()
def render_field(field: Callable[[torch.Tensor], torch.Tensor], to_field: np.ndarray, T: np.ndarray,
                 size: tuple[int, int], alpha: np.ndarray, device: torch.device,
                 chunk: int = 1 << 16) -> np.ndarray:
    """Render a neural image: output pixel -> reference pixel (T^-1) -> field coordinates."""
    w, h = size
    M = torch.tensor(to_field @ np.linalg.inv(T), dtype=torch.float32, device=device)
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2)
    uv = pix @ M[:2, :2].T + M[:2, 2]  # T and the canvas normalisation are affine
    out = torch.cat([field(uv[k:k + chunk]) for k in range(0, len(uv), chunk)])
    rgb = (out.clamp(0, 1) * 255).round().byte().reshape(h, w, 3).cpu().numpy()
    rgb[alpha == 0] = 0  # the network invents colours where no image is; drop them like the classic blend
    return np.dstack([rgb, alpha])
