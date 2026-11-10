import itertools

import numpy as np

from neural_stitch.bundle import refine_homographies
from neural_stitch.config import BundleConfig
from neural_stitch.features import PairMatch
from neural_stitch.geometry import apply_h
from neural_stitch.graph import build_report

from .conftest import corner_error, view_transform

W, H = 320, 240
GS = [view_transform(0, 0), view_transform(200, 10, 0.03), view_transform(400, -5, -0.02, 1.02),
      view_transform(600, 20, 0.01, 0.98, 1e-5)]


def exact_pair(i: int, j: int, Gs: list[np.ndarray], n_pts: int = 0, rng=None) -> PairMatch:
    Hij = np.linalg.inv(Gs[i]) @ Gs[j]  # j -> i
    p = PairMatch(i=i, j=j, num_matches=200, num_inliers=200, H=Hij / Hij[2, 2])
    if n_pts:
        pts_j = rng.random((n_pts, 2)) * [W, H]
        p.pts_j, p.pts_i = pts_j, apply_h(Hij, pts_j)
    return p


def test_chain_follows_strongest_matches_and_is_correct():
    # chain 0-1-2-3; image 3 only reachable through 2
    pairs = [exact_pair(0, 1, GS), exact_pair(1, 2, GS), exact_pair(2, 3, GS)]
    pairs.append(PairMatch(i=0, j=3, num_matches=5, reason="only 5 matches"))
    report, Hs = build_report(["a", "b", "c", "d"], pairs)
    assert report.ok and sorted(Hs) == [0, 1, 2, 3]
    ref = report.reference
    assert ref in (1, 2)  # best connected
    for k in Hs:
        truth = np.linalg.inv(GS[ref]) @ GS[k]
        assert corner_error(Hs[k], truth, W, H) < 1e-6


def test_unmatched_image_is_left_out_and_order_does_not_matter():
    names = ["a", "b", "c", "x"]
    pairs = [exact_pair(0, 1, GS), exact_pair(1, 2, GS), exact_pair(0, 2, GS),
             PairMatch(i=0, j=3, num_matches=3, reason="only 3 matches")]
    report, _ = build_report(names, pairs)
    assert [im.used for im in report.images] == [True, True, True, False]
    assert report.images[3].reason == "no reliable match with any other image"

    # Same scene, images given in a different order -> same set of used images.
    for perm in itertools.permutations(range(4)):
        inv = {old: new for new, old in enumerate(perm)}
        p2 = []
        for p in pairs:
            a, b = sorted((inv[p.i], inv[p.j]))
            p2.append(PairMatch(i=a, j=b, num_matches=p.num_matches, num_inliers=p.num_inliers,
                                H=np.eye(3), reason=p.reason))
        r2, _ = build_report([names[k] for k in perm], p2)
        assert {im.name for im in r2.images if im.used} == {"a", "b", "c"}


def test_nothing_alignable():
    report, Hs = build_report(["a", "b"], [PairMatch(i=0, j=1, num_matches=2, reason="only 2 matches")])
    assert not report.ok and Hs == {}
    report, _ = build_report(["a"], [])
    assert not report.ok and "at least 2" in report.message


def test_least_squares_fixes_noisy_start():
    rng = np.random.default_rng(0)
    Gs = GS[:3]
    pairs = [exact_pair(i, j, Gs, 300, rng) for i, j in [(0, 1), (1, 2), (0, 2)]]
    for p in pairs:  # 0.3 px keypoint noise
        p.pts_i = p.pts_i + rng.normal(0, 0.3, p.pts_i.shape)
    truth = {k: np.linalg.inv(Gs[0]) @ Gs[k] for k in range(3)}
    noisy = {0: np.eye(3)}
    for k in (1, 2):
        D = np.eye(3) + rng.normal(0, 1e-2, (3, 3)) * [[1, 1, 100], [1, 1, 100], [1e-4, 1e-4, 0]]
        noisy[k] = truth[k] @ D
    refined, stats = refine_homographies(noisy, {k: (W, H) for k in range(3)}, pairs, 0, BundleConfig())
    assert stats["rms_after_px"] < stats["rms_before_px"]
    for k in (1, 2):
        assert corner_error(refined[k], truth[k], W, H) < 0.15
        assert corner_error(noisy[k], truth[k], W, H) > 0.5
    assert np.allclose(refined[0], np.eye(3))  # reference stays fixed
