"""Match graph: which images belong together, which one is the reference, and their start poses.

Images are nodes; an accepted pair match is an edge weighted by its number of inliers. The input
order does not matter: the largest connected group is kept and its best-connected image becomes
the reference. Start poses are chained along a maximum spanning tree (strongest matches first).
"""
import heapq

import numpy as np
from pydantic import BaseModel

from .features import PairMatch
from .geometry import normalize_det


class ImageStatus(BaseModel):
    index: int
    name: str
    used: bool
    reference: bool = False
    reason: str | None = None
    neighbors: list[int] = []


class PairSummary(BaseModel):
    i: int
    j: int
    matches: int
    inliers: int
    ok: bool
    reason: str | None = None


class AlignmentReport(BaseModel):
    ok: bool
    message: str
    reference: int | None
    images: list[ImageStatus]
    pairs: list[PairSummary]
    groups: list[list[int]]


def _neighbors(n: int, pairs: list[PairMatch]) -> list[dict[int, int]]:
    adj: list[dict[int, int]] = [{} for _ in range(n)]
    for p in pairs:
        if p.ok:
            adj[p.i][p.j] = p.num_inliers
            adj[p.j][p.i] = p.num_inliers
    return adj


def connected_groups(adj: list[dict[int, int]]) -> list[list[int]]:
    """Connected components, largest (then strongest) first."""
    seen, groups = set(), []
    for start in range(len(adj)):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            k = stack.pop()
            group.append(k)
            for m in adj[k]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        groups.append(sorted(group))

    def strength(g: list[int]) -> tuple[int, int]:
        return len(g), sum(sum(adj[k].values()) for k in g)

    return sorted(groups, key=strength, reverse=True)


def spanning_tree(adj: list[dict[int, int]], root: int) -> dict[int, int]:
    """Maximum spanning tree (Prim) from root. Returns child -> parent."""
    parent: dict[int, int] = {}
    done = {root}
    heap = [(-w, root, m) for m, w in adj[root].items()]
    heapq.heapify(heap)
    while heap:
        _, p, c = heapq.heappop(heap)
        if c in done:
            continue
        done.add(c)
        parent[c] = p
        for m, w in adj[c].items():
            if m not in done:
                heapq.heappush(heap, (-w, c, m))
    return parent


def pair_homography(pairs: dict[tuple[int, int], PairMatch], src: int, dst: int) -> np.ndarray:
    """Homography mapping image src -> image dst (pairs are stored with i < j, H: j -> i)."""
    if src > dst:
        return pairs[(dst, src)].H
    return np.linalg.inv(pairs[(src, dst)].H)


def chain_homographies(pairs: dict[tuple[int, int], PairMatch], root: int,
                       parent: dict[int, int]) -> dict[int, np.ndarray]:
    """Start pose of every tree node: maps its pixels to the reference image pixels."""
    H = {root: np.eye(3)}

    def solve(k: int) -> np.ndarray:
        if k not in H:
            p = parent[k]
            H[k] = normalize_det(solve(p) @ pair_homography(pairs, k, p))
        return H[k]

    for k in parent:
        solve(k)
    return H


def build_report(names: list[str], pairs: list[PairMatch]) -> tuple[AlignmentReport, dict[int, np.ndarray]]:
    """Decide which images can be stitched and compute their start poses."""
    n = len(names)
    adj = _neighbors(n, pairs)
    groups = connected_groups(adj) if n else []
    summaries = [PairSummary(i=p.i, j=p.j, matches=p.num_matches, inliers=p.num_inliers,
                             ok=p.ok, reason=p.reason) for p in pairs]
    main = groups[0] if groups and len(groups[0]) >= 2 else []

    if not main:
        msg = ("Need at least 2 images." if n < 2
               else "No two images overlap reliably, so nothing can be stitched.")
        images = [ImageStatus(index=k, name=names[k], used=False,
                              reason="no reliable match with any other image") for k in range(n)]
        return AlignmentReport(ok=False, message=msg, reference=None, images=images,
                               pairs=summaries, groups=groups), {}

    ref = max(main, key=lambda k: (sum(adj[k].values()), -k))
    by_key = {(p.i, p.j): p for p in pairs if p.ok}
    H = chain_homographies(by_key, ref, spanning_tree(adj, ref))

    images = []
    for k in range(n):
        if k in H:
            images.append(ImageStatus(index=k, name=names[k], used=True, reference=k == ref,
                                      neighbors=sorted(adj[k])))
        else:
            reason = ("no reliable match with any other image" if not adj[k]
                      else "only matches images outside the main group")
            images.append(ImageStatus(index=k, name=names[k], used=False, reason=reason,
                                      neighbors=sorted(adj[k])))
    left_out = n - len(main)
    msg = f"{len(main)} of {n} images can be stitched."
    if left_out:
        msg += f" {left_out} image{'s' if left_out > 1 else ''} left out."
    return AlignmentReport(ok=True, message=msg, reference=ref, images=images,
                           pairs=summaries, groups=groups), H
