"""Keypoints, matching and the per-pair RANSAC check.

Features are computed once per image (not once per pair) on a shrunk copy of the image.
"""
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import kornia.feature as KF
import numpy as np
import torch

from .config import MatchConfig
from .geometry import check_warp, scale_matrix
from .images import InputImage, resize_max


@dataclass
class Features:
    keypoints: np.ndarray  # [K, 2] in full-resolution pixels
    match_keypoints: Any  # backend-specific keypoints at matching resolution
    descriptors: Any
    match_hw: tuple[int, int]
    scale: float  # matching pixels = scale * full pixels


@dataclass
class PairMatch:
    i: int
    j: int
    num_matches: int
    num_inliers: int = 0
    H: np.ndarray | None = None  # maps image j -> image i (full-resolution pixels)
    pts_i: np.ndarray | None = None  # RANSAC inliers [K, 2], full-resolution pixels
    pts_j: np.ndarray | None = None
    reason: str | None = None  # why the pair was rejected (None = accepted)

    @property
    def ok(self) -> bool:
        return self.reason is None

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / self.num_matches if self.num_matches else 0.0


class Backend(Protocol):
    def extract(self, rgb: np.ndarray, num_features: int) -> tuple[Any, Any]: ...
    def match(self, a: Features, b: Features) -> np.ndarray: ...
    def to_numpy(self, keypoints: Any) -> np.ndarray: ...


class DiskLightGlue:
    """DISK keypoints + LightGlue matching (learned, robust). Weights are downloaded on first use."""

    def __init__(self, device: torch.device):
        self.device = device
        self.disk = KF.DISK.from_pretrained("depth", device=device).eval()
        self.matcher = KF.LightGlueMatcher("disk").eval().to(device)

    @torch.inference_mode()
    def extract(self, rgb: np.ndarray, num_features: int) -> tuple[Any, Any]:
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float()[None] / 255
        f = self.disk(x, num_features, pad_if_not_divisible=True)[0]
        return f.keypoints, f.descriptors

    @torch.inference_mode()
    def match(self, a: Features, b: Features) -> np.ndarray:
        if len(a.match_keypoints) == 0 or len(b.match_keypoints) == 0:
            return np.zeros((0, 2), dtype=np.int64)

        def lafs(k: torch.Tensor) -> torch.Tensor:
            return KF.laf_from_center_scale_ori(k[None], torch.ones(1, len(k), 1, 1, device=self.device))

        _, idx = self.matcher(a.descriptors, b.descriptors, lafs(a.match_keypoints),
                              lafs(b.match_keypoints), hw1=a.match_hw, hw2=b.match_hw)
        return idx.cpu().numpy()

    def to_numpy(self, keypoints: Any) -> np.ndarray:
        return keypoints.cpu().numpy().astype(np.float64)


class Sift:
    """OpenCV SIFT + ratio test. No downloads, CPU only; used for tests and as a fallback."""

    def __init__(self, ratio: float = 0.8):
        self.ratio = ratio

    def extract(self, rgb: np.ndarray, num_features: int) -> tuple[Any, Any]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, desc = cv2.SIFT_create(nfeatures=num_features).detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], dtype=np.float64).reshape(-1, 2)
        return pts, desc

    def match(self, a: Features, b: Features) -> np.ndarray:
        if a.descriptors is None or b.descriptors is None or len(a.descriptors) < 2 or len(b.descriptors) < 2:
            return np.zeros((0, 2), dtype=np.int64)
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.descriptors, b.descriptors, k=2)
        good = [(m.queryIdx, m.trainIdx) for m, n in (p for p in knn if len(p) == 2)
                if m.distance < self.ratio * n.distance]
        return np.array(good, dtype=np.int64).reshape(-1, 2)

    def to_numpy(self, keypoints: Any) -> np.ndarray:
        return keypoints


def make_backend(cfg: MatchConfig, device: torch.device) -> Backend:
    if cfg.backend == "sift":
        return Sift()
    return DiskLightGlue(device)


def extract_features(image: InputImage, backend: Backend, cfg: MatchConfig) -> Features:
    small, s = resize_max(image.rgb, cfg.max_side)
    kps, desc = backend.extract(small, cfg.num_features)
    # Back to full-resolution pixels.
    full = np.linalg.inv(scale_matrix(s))
    pts = backend.to_numpy(kps)
    pts_full = pts * full[0, 0] + full[0, 2] if len(pts) else pts.reshape(0, 2)
    return Features(keypoints=pts_full, match_keypoints=kps, descriptors=desc,
                    match_hw=small.shape[:2], scale=s)


def match_pair(i: int, j: int, fi: Features, fj: Features, size_j: tuple[int, int],
               backend: Backend, cfg: MatchConfig) -> PairMatch:
    """Match two images and decide whether they reliably overlap."""
    idx = backend.match(fi, fj)
    pair = PairMatch(i=i, j=j, num_matches=len(idx))
    if len(idx) < cfg.min_inliers:
        pair.reason = f"only {len(idx)} matches"
        return pair
    pi, pj = fi.keypoints[idx[:, 0]], fj.keypoints[idx[:, 1]]
    H, mask = cv2.findHomography(pj, pi, cv2.USAC_MAGSAC, cfg.ransac_threshold / fi.scale, 0.999, 10000)
    if H is None:
        pair.reason = "RANSAC found no homography"
        return pair
    inl = mask.ravel().astype(bool)
    pair.num_inliers = int(inl.sum())
    if pair.num_inliers < cfg.min_inliers:
        pair.reason = f"only {pair.num_inliers} consistent matches"
        return pair
    if pair.inlier_ratio < cfg.min_inlier_ratio:
        pair.reason = f"only {pair.inlier_ratio:.0%} of the matches agree"
        return pair
    reason = check_warp(H, *size_j, max_area_change=cfg.max_area_change)
    if reason:
        pair.reason = reason
        return pair
    pair.H = H / H[2, 2]
    pair.pts_i, pair.pts_j = pi[inl], pj[inl]
    return pair
