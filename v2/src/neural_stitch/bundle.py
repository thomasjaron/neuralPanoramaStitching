"""Least-squares refinement ("bundle adjustment") of all homographies on the keypoint matches.

Uses every accepted pair, not only the spanning-tree edges, so errors that pile up along a chain
get spread out. The reference image stays fixed. Each homography is updated as
``H = H0 @ N^-1 @ exp(p) @ N`` with N normalising the image to [-1, 1], so the 8 parameters are
well scaled. The error of a match is measured in image pixels in both directions, so the solver
cannot cheat by shrinking images.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .config import BundleConfig
from .features import PairMatch
from .geometry import apply_h_t, normalize_det, normalizer, sl3_exp


@dataclass
class MatchData:
    """All inlier matches stacked: match m belongs to pair pair_id[m] = (a[pair], b[pair])."""

    a: torch.Tensor  # [P] image position of pair side i
    b: torch.Tensor  # [P] image position of pair side j
    pair_id: torch.Tensor  # [M]
    xa: torch.Tensor  # [M, 2]
    xb: torch.Tensor  # [M, 2]


def stack_matches(pairs: list[PairMatch], pos: dict[int, int], device: torch.device,
                  dtype: torch.dtype) -> MatchData:
    a, b, pid, xa, xb = [], [], [], [], []
    for k, p in enumerate(pairs):
        a.append(pos[p.i])
        b.append(pos[p.j])
        pid.append(np.full(len(p.pts_i), k))
        xa.append(p.pts_i)
        xb.append(p.pts_j)

    def t(x: list, dt: torch.dtype = dtype) -> torch.Tensor:
        return torch.as_tensor(np.concatenate(x) if isinstance(x[0], np.ndarray) else np.array(x),
                               dtype=dt, device=device)

    return MatchData(a=t(a, torch.long), b=t(b, torch.long), pair_id=t(pid, torch.long), xa=t(xa), xb=t(xb))


def transfer_residuals(H: torch.Tensor, m: MatchData) -> torch.Tensor:
    """Pixel errors [2M, 2]: b's point mapped into a vs a's point, and the other way round.

    H[k] maps image k into any common frame; that frame cancels out.
    """
    Hab = torch.linalg.solve(H[m.a], H[m.b])  # maps b -> a, [P, 3, 3]
    Hba = torch.linalg.inv(Hab)
    r1 = apply_h_t(Hab[m.pair_id], m.xb[:, None])[:, 0] - m.xa
    r2 = apply_h_t(Hba[m.pair_id], m.xa[:, None])[:, 0] - m.xb
    return torch.cat([r1, r2])


def rms(r: torch.Tensor) -> float:
    return float(r.pow(2).sum(-1).mean().sqrt())


def refine_homographies(H_init: dict[int, np.ndarray], sizes: dict[int, tuple[int, int]],
                        pairs: list[PairMatch], ref: int,
                        cfg: BundleConfig) -> tuple[dict[int, np.ndarray], dict]:
    idxs = sorted(H_init)
    pos = {k: n for n, k in enumerate(idxs)}
    pairs = [p for p in pairs if p.ok and p.i in pos and p.j in pos]
    dt = torch.float64
    H0 = torch.tensor(np.stack([H_init[k] for k in idxs]), dtype=dt)
    N = torch.tensor(np.stack([normalizer(*sizes[k]) for k in idxs]), dtype=dt)
    Ninv = torch.linalg.inv(N)
    mask = torch.ones(len(idxs), 1, dtype=dt)
    mask[pos[ref]] = 0
    m = stack_matches(pairs, pos, torch.device("cpu"), dt)
    params = torch.zeros(len(idxs), 8, dtype=dt, requires_grad=True)

    def current() -> torch.Tensor:
        return H0 @ Ninv @ sl3_exp(params * mask) @ N

    c2 = cfg.robust_scale ** 2

    def closure() -> torch.Tensor:
        opt.zero_grad()
        e = transfer_residuals(current(), m).pow(2).sum(-1)
        loss = (c2 * torch.log1p(e / c2)).mean()  # Cauchy: big outliers count less
        loss.backward()
        return loss

    with torch.no_grad():
        before = rms(transfer_residuals(H0, m))
    opt = torch.optim.LBFGS([params], lr=1, max_iter=cfg.max_iters, tolerance_grad=1e-12,
                            tolerance_change=1e-14, history_size=20, line_search_fn="strong_wolfe")
    opt.step(closure)
    with torch.no_grad():
        Hn = current()
        after = rms(transfer_residuals(Hn, m))
    out = {k: normalize_det(Hn[pos[k]].numpy()) for k in idxs}
    return out, {"rms_before_px": before, "rms_after_px": after, "matches": int(len(m.xa))}
