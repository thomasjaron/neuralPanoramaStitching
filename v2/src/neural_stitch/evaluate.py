"""Compare results with SIDAR ground truth (``homography/H_0_<b>.mat`` maps image 0 -> image b).

Image 0 is the full ground-truth scene, so it should not be an input (it would hide misalignments
and act as an anchor). Errors are measured relative to one stitched image (the anchor) and
expressed in image-0 pixels, so results with and without image 0 are comparable.
"""
from pathlib import Path

import numpy as np

from .geometry import apply_h, corners
from .images import InputImage

GT_NAME = "0"


def load_sidar_truth(data_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    """Ground truth homographies image b -> image 0, keyed by image name."""
    truth = {GT_NAME: np.eye(3)}
    for name in names:
        f = Path(data_dir) / "homography" / f"H_0_{name}.mat"
        if f.exists():
            truth[name] = np.linalg.inv(np.loadtxt(f))
    return truth


def corner_errors(H: dict[int, np.ndarray], images: list[InputImage], truth: dict[str, np.ndarray],
                  anchor: int) -> dict[str, float]:
    """Mean corner distance (image-0 pixels) of every stitched image except the anchor.

    The estimate is placed in image 0 through the anchor's true pose:
    ``b -> 0 = truth[anchor] @ H[anchor]^-1 @ H[b]``.
    """
    a = images[anchor].name
    if a not in truth:
        raise ValueError(f"No ground truth for the anchor image {a}.")
    place = truth[a] @ np.linalg.inv(H[anchor])
    errors = {}
    for k, Hk in H.items():
        name = images[k].name
        if k == anchor or name not in truth:
            continue
        c = corners(*images[k].size)
        errors[name] = float(np.linalg.norm(apply_h(place @ Hk, c) - apply_h(truth[name], c), axis=1).mean())
    return errors


def pairwise_error(H: dict[int, np.ndarray], images: list[InputImage], truth: dict[str, np.ndarray]) -> float:
    """Mean corner error over every image used as the anchor, i.e. over all image pairs.

    Unlike ``corner_errors`` this doesn't favour pairs that include the reference image.
    """
    errs = []
    for anchor in H:
        if images[anchor].name in truth:
            errs += list(corner_errors(H, images, truth, anchor).values())
    return float(np.mean(errs))
