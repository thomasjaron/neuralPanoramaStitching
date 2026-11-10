"""Neural refinement: jointly learn a neural image of the panorama and small homography corrections.

Starting point is the least-squares result. Every step samples small patches from all images, maps
them onto the canvas with the current homographies and compares their colours (and colour
gradients, which hides exposure differences) with the neural image. A keypoint term keeps the
geometry anchored. Training stops when the loss and the homographies stop changing.
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .bundle import stack_matches, transfer_residuals
from .config import NeuralConfig
from .features import PairMatch
from .geometry import apply_h, apply_h_t, corners, normalize_det, normalizer, scale_matrix, sl3_exp
from .images import resize_max


class Cancelled(Exception):
    pass


class NeuralImage(nn.Module):
    """MLP from canvas coordinates [-1, 1] to RGB, with coarse-to-fine positional encoding (BARF)."""

    def __init__(self, freqs: int, hidden: int, layers: int):
        super().__init__()
        self.freqs = freqs
        dims = [2 + 4 * freqs] + [hidden] * layers + [3]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True))
        self.register_buffer("bands", (2.0 ** torch.arange(freqs)) * math.pi)
        self.alpha = float(freqs)  # how many frequency bands are switched on

    def encode(self, uv: torch.Tensor) -> torch.Tensor:
        if self.freqs == 0:
            return uv
        k = torch.arange(self.freqs, device=uv.device, dtype=uv.dtype)
        weight = (1 - torch.cos((self.alpha - k).clamp(0, 1) * math.pi)) / 2  # [L]
        s = uv[..., None] * self.bands  # [M, 2, L]
        enc = torch.cat([s.sin() * weight, s.cos() * weight], -1)
        return torch.cat([uv, enc.flatten(-2)], -1)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        h = self.encode(uv)
        for layer in self.layers[:-1]:
            h = torch.relu(layer(h))
        return torch.sigmoid(self.layers[-1](h))


@dataclass
class NeuralResult:
    homographies: dict[int, np.ndarray]  # image -> reference pixels
    model: NeuralImage
    to_field: np.ndarray  # reference pixels -> neural image coordinates
    iterations: int
    stop_reason: str
    history: list[dict] = field(default_factory=list)
    rms_px: float = 0.0


@dataclass
class ConvergenceMonitor:
    """Decides when training has converged, from one mean loss per check window.

    - Trend: the improvement is the slope of log(loss) over the last ``trend_windows`` windows
      (relative improvement per window). A single noisy window can't fake or hide a plateau.
    - A plateau (trend below ``rel_tol``) halves the learning rates. A window that still improves
      lowers the plateau count by one instead of resetting it, so progress isn't thrown away.
    - Stop after ``patience`` plateaus once the homographies also move less than ``corner_tol`` px.
    """

    cfg: NeuralConfig
    losses: list[float] = field(default_factory=list)
    plateaus: int = 0
    lr_scale: float = 1.0

    def update(self, it: int, loss: float, motion: float) -> tuple[float | None, bool]:
        """Record one window. Returns (trend, stop)."""
        self.losses.append(loss)
        trend = None
        if len(self.losses) >= 2:
            y = np.log(self.losses[-self.cfg.trend_windows:])
            trend = float(-np.polyfit(np.arange(len(y)), y, 1)[0])
        if it >= self.cfg.min_iters and trend is not None and trend < self.cfg.rel_tol:
            self.plateaus += 1
            self.lr_scale = max(self.lr_scale / 2, 1 / 64)
        else:
            self.plateaus = max(0, self.plateaus - 1)
        return trend, self.plateaus >= self.cfg.patience and motion < self.cfg.corner_tol


def canvas_normalizer(Hs: list[np.ndarray], sizes: list[tuple[int, int]]) -> np.ndarray:
    """Reference pixels -> [-1, 1] over the whole canvas (long side), with a small margin."""
    pts = np.concatenate([apply_h(H, corners(w, h)) for H, (w, h) in zip(Hs, sizes, strict=True)])
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    r = max(hi - lo) / 2 * 1.05
    return np.array([[1 / r, 0, -c[0] / r], [0, 1 / r, -c[1] / r], [0, 0, 1]])


def refine_neural(images: dict[int, np.ndarray], H_init: dict[int, np.ndarray], pairs: list[PairMatch],
                  ref: int, cfg: NeuralConfig, device: torch.device,
                  on_check: Callable[[dict, Callable[[], np.ndarray] | None], None] | None = None,
                  should_cancel: Callable[[], bool] | None = None,
                  preview: Callable[[NeuralImage, np.ndarray, dict[int, np.ndarray]], np.ndarray] | None = None,
                  seed: int = 0) -> NeuralResult:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    idxs = sorted(H_init)
    pos = {k: n for n, k in enumerate(idxs)}
    n = len(idxs)
    sizes = [(images[k].shape[1], images[k].shape[0]) for k in idxs]
    C = canvas_normalizer([H_init[k] for k in idxs], sizes)
    C_inv = np.linalg.inv(C)

    f32 = dict(dtype=torch.float32, device=device)
    work, to_full = [], []
    for k in idxs:
        small, s = resize_max(images[k], cfg.work_side)
        work.append(torch.as_tensor(small, device=device).float() / 255)
        to_full.append(np.linalg.inv(scale_matrix(s)))
    H0 = torch.tensor(np.stack([C @ H_init[k] for k in idxs]), **f32)  # full pixels -> canvas
    N = torch.tensor(np.stack([normalizer(*sz) for sz in sizes]), **f32)
    N_inv = torch.linalg.inv(N)
    work_to_full = torch.tensor(np.stack(to_full), **f32)
    mask = torch.ones(n, 1, **f32)
    mask[pos[ref]] = 0

    model = NeuralImage(cfg.freqs, cfg.hidden, cfg.layers).to(device)
    delta = torch.zeros(n, 8, **f32, requires_grad=True)
    gain = torch.zeros(n, 3, **f32, requires_grad=True)  # per-image exposure: colour * exp(gain) + bias
    bias = torch.zeros(n, 3, **f32, requires_grad=True)
    base_lrs = [cfg.lr_image, cfg.lr_warp, cfg.lr_color]
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": base_lrs[0]},
                            {"params": [delta], "lr": base_lrs[1]},
                            {"params": [gain, bias], "lr": base_lrs[2]}])
    matches = stack_matches(pairs, pos, device, torch.float32) if pairs else None

    k = cfg.patch_size
    oy, ox = torch.meshgrid(torch.arange(k, device=device), torch.arange(k, device=device), indexing="ij")

    def current_H() -> torch.Tensor:
        return H0 @ N_inv @ sl3_exp(delta * mask) @ N

    def ref_homographies() -> dict[int, np.ndarray]:
        with torch.no_grad():
            Hc = current_H().double().cpu().numpy()
        return {kk: normalize_det(C_inv @ Hc[pos[kk]]) for kk in idxs}

    def corner_positions() -> np.ndarray:
        Hs = ref_homographies()
        return np.stack([apply_h(Hs[kk], corners(*sizes[pos[kk]])) for kk in idxs])

    monitor = ConvergenceMonitor(cfg)
    window: list[float] = []
    prev_corners = corner_positions()
    history: list[dict] = []
    stop_reason = "max_iters"
    it = 0
    for it in range(1, cfg.max_iters + 1):
        if should_cancel and should_cancel():
            raise Cancelled()
        model.alpha = cfg.freqs * min(1.0, it / cfg.c2f_iters) if cfg.c2f_iters else float(cfg.freqs)
        Hc = current_H()
        M = Hc @ work_to_full  # work pixels -> canvas

        uv, targets, owner = [], [], []
        for p, img in enumerate(work):
            h, w = img.shape[:2]
            kh, kw = min(k, h), min(k, w)
            x0 = torch.randint(0, w - kw + 1, (cfg.patches_per_image, 1, 1), device=device, generator=gen)
            y0 = torch.randint(0, h - kh + 1, (cfg.patches_per_image, 1, 1), device=device, generator=gen)
            xs, ys = x0 + ox[:kh, :kw], y0 + oy[:kh, :kw]  # [P, k, k]
            targets.append(img[ys, xs])
            pix = torch.stack([xs, ys], -1).float().reshape(-1, 2)
            uv.append(apply_h_t(M[p], pix))
            owner.append((p, xs.shape))
        pred_all = model(torch.cat(uv))

        loss_rgb = torch.zeros((), **f32)
        loss_grad = torch.zeros((), **f32)
        start = 0
        for (p, shape), tgt in zip(owner, targets, strict=True):
            cnt = shape.numel()
            g = torch.where(mask[p] > 0, gain[p], torch.zeros_like(gain[p]))
            b = bias[p] * mask[p]
            pred = (pred_all[start:start + cnt].reshape(*shape, 3) * g.exp() + b)
            start += cnt
            loss_rgb = loss_rgb + (pred - tgt).pow(2).mean()
            loss_grad = loss_grad + ((pred.diff(dim=1) - tgt.diff(dim=1)).abs().mean()
                                     + (pred.diff(dim=2) - tgt.diff(dim=2)).abs().mean())
        loss_rgb, loss_grad = loss_rgb / n, loss_grad / n
        loss = cfg.weight_rgb * loss_rgb + cfg.weight_grad * loss_grad
        loss_kp = torch.zeros((), **f32)
        if matches is not None and cfg.weight_keypoints > 0:
            loss_kp = transfer_residuals(Hc, matches).pow(2).sum(-1).mean()
            loss = loss + cfg.weight_keypoints * loss_kp

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if it <= cfg.warp_warmup and delta.grad is not None:
            delta.grad.zero_()
        opt.step()
        window.append(float((loss_rgb + cfg.weight_grad * loss_grad).detach()))

        if it % cfg.check_every == 0:
            cur = float(np.mean(window))
            window.clear()
            cp = corner_positions()
            motion = float(np.abs(cp - prev_corners).max())
            prev_corners = cp
            trend, stop = monitor.update(it, cur, motion)
            for group, lr in zip(opt.param_groups, base_lrs, strict=True):
                group["lr"] = lr * monitor.lr_scale
            entry = {"iter": it, "loss": cur, "rel_improvement": trend,
                     "corner_motion_px": motion, "keypoint_rms_px": float(loss_kp.detach().sqrt()),
                     "lr_scale": monitor.lr_scale, "plateaus": monitor.plateaus}
            history.append(entry)
            checks = len(history)
            make_preview = None
            if preview and cfg.preview_every and checks % cfg.preview_every == 0:
                def make_preview() -> np.ndarray:
                    return preview(model, C, ref_homographies())
            if on_check:
                on_check(entry, make_preview)
            if stop:
                stop_reason = "converged"
                break

    final = ref_homographies()
    rms_px = 0.0
    if matches is not None:
        with torch.no_grad():
            Ht = torch.tensor(np.stack([final[kk] for kk in idxs]), **f32)
            rms_px = float(transfer_residuals(Ht, matches).pow(2).sum(-1).mean().sqrt())
    model.alpha = float(cfg.freqs)
    return NeuralResult(homographies=final, model=model.eval(), to_field=C, iterations=it,
                        stop_reason=stop_reason, history=history, rms_px=rms_px)
