"""All settings in one place, with defaults and validation."""
from typing import Literal

import torch
from pydantic import BaseModel, Field


class MatchConfig(BaseModel):
    """Feature extraction, matching and the per-pair RANSAC check."""

    backend: Literal["disk-lightglue", "sift"] = "disk-lightglue"
    max_side: int = Field(1024, ge=128, description="Images are shrunk to this long side for matching.")
    num_features: int = Field(2048, ge=64)
    ransac_threshold: float = Field(2.0, gt=0, description="RANSAC inlier distance, in matching pixels.")
    min_inliers: int = Field(40, ge=8, description="A pair needs this many RANSAC inliers to count as a match.")
    min_inlier_ratio: float = Field(0.15, ge=0, le=1)
    max_area_change: float = Field(8.0, gt=1, description="Reject warps that grow/shrink an image more.")


# Matching speed presets. Measured on 5 photo-sized images (CPU): "fast" checks ~2.3x faster with
# about the same accuracy on that data; real photos with fine detail may lose a little more.
MATCH_PRESETS: dict[str, dict] = {
    "accurate": {"max_side": 1024, "num_features": 2048},
    "fast": {"max_side": 768, "num_features": 1024},
}


def match_preset(base: MatchConfig, speed: str) -> MatchConfig:
    """``base`` with the image size and keypoint count of a speed preset."""
    if speed not in MATCH_PRESETS:
        raise ValueError(f"unknown speed '{speed}', use one of {list(MATCH_PRESETS)}")
    return base.model_copy(update=MATCH_PRESETS[speed])


class BundleConfig(BaseModel):
    """Least-squares refinement of all homographies on the keypoint matches."""

    max_iters: int = Field(200, ge=1)
    robust_scale: float = Field(3.0, gt=0, description="Cauchy loss scale in full-resolution pixels.")


class NeuralConfig(BaseModel):
    """Joint refinement of homographies and a neural image, until the error converges."""

    enabled: bool = True
    work_side: int = Field(512, ge=32, description="Images are shrunk to this long side for training.")
    hidden: int = Field(256, ge=8)
    layers: int = Field(4, ge=1)
    freqs: int = Field(10, ge=0, description="Number of positional-encoding frequency bands.")
    c2f_iters: int = Field(800, ge=0, description="Frequency bands are switched on over this many steps.")
    warp_warmup: int = Field(200, ge=0, description="Homographies stay fixed for the first steps.")
    patch_size: int = Field(8, ge=2)
    patches_per_image: int = Field(64, ge=1)
    lr_image: float = 1e-3
    lr_warp: float = 1e-4
    lr_color: float = 1e-3
    weight_rgb: float = 1.0
    weight_grad: float = 0.5
    weight_keypoints: float = 1e-3
    # Convergence (see neural.ConvergenceMonitor): every `check_every` steps the mean loss of the
    # window is recorded. The improvement is the trend over the last `trend_windows` windows; below
    # `rel_tol` it counts as a plateau and halves the learning rates, a better window lowers the
    # count by one. Training stops after `patience` plateaus once the homographies also moved less
    # than `corner_tol` pixels in the last window.
    min_iters: int = Field(1000, ge=0)
    max_iters: int = Field(6000, ge=1)
    check_every: int = Field(100, ge=1)
    trend_windows: int = Field(4, ge=2)
    rel_tol: float = Field(5e-3, ge=0)
    corner_tol: float = Field(0.1, ge=0, description="Max corner movement in pixels per check window.")
    patience: int = Field(3, ge=1)
    preview_every: int = Field(5, ge=0, description="Send a preview every N checks (0 = never).")
    preview_side: int = Field(512, ge=64)


# Video-style presets: the number is the short side (height for landscape panoramas).
RESOLUTIONS: dict[str, tuple[int, str]] = {
    "144p": (144, ""),
    "240p": (240, ""),
    "360p": (360, ""),
    "480p": (480, "SD"),
    "720p": (720, "HD"),
    "1080p": (1080, "Full HD"),
    "1440p": (1440, "QHD"),
    "2160p": (2160, "4K UHD"),
    "4320p": (4320, "8K UHD"),
}
Resolution = Literal["native", "144p", "240p", "360p", "480p", "720p", "1080p", "1440p", "2160p", "4320p"]


class OutputConfig(BaseModel):
    resolution: Resolution = Field("native", description=(
        "'native' = the images' own resolution, limited by max_side. A preset like '1080p' sets the short "
        "side (height of a landscape panorama); the width follows the panorama's shape."))
    max_side: int = Field(4000, ge=64, description="Long-side limit in pixels for 'native'.")
    max_megapixels: float = Field(50.0, gt=0, description="Safety limit for every mode (memory).")


class StitchConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 0
    max_input_side: int = Field(3000, ge=64, description="Larger uploads are shrunk on load.")
    match: MatchConfig = MatchConfig()
    bundle: BundleConfig = BundleConfig()
    neural: NeuralConfig = NeuralConfig()
    output: OutputConfig = OutputConfig()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
