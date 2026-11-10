import numpy as np
import pytest

from neural_stitch.pipeline import Cancelled, Stitcher

from .conftest import corner_error


@pytest.fixture(scope="module")
def checked(scene, request):
    from neural_stitch.config import StitchConfig
    cfg = StitchConfig(device="cpu")
    cfg.match.backend = "sift"
    cfg.match.min_inliers = 30
    images, _ = scene
    stitcher = Stitcher(cfg)
    return stitcher, stitcher.check(images)


def test_check_finds_the_group_and_leaves_out_the_stranger(checked, scene):
    _, check = checked
    r = check.report
    assert r.ok
    assert [im.used for im in r.images] == [True, False, True, True]
    assert r.images[1].reason == "no reliable match with any other image"
    assert check.preview is not None and check.preview.shape[2] == 4


def test_homographies_match_ground_truth(checked, scene):
    _, check = checked
    _, truth = scene
    ref = check.report.reference
    for k, Hk in check.homographies.items():
        expected = np.linalg.inv(truth[ref]) @ truth[k]
        assert corner_error(Hk, expected, 320, 240) < 0.5, k
    assert check.metrics["keypoint_rms_bundle_px"] <= check.metrics["keypoint_rms_ransac_px"] + 1e-9


def test_stitch_reports_progress_and_renders(scene, fast_config, checked):
    images, truth = scene
    _, check = checked
    events = []
    result = Stitcher(fast_config).stitch(images, check, on_progress=events.append)
    assert result.panorama.ndim == 3 and result.panorama.shape[2] == 4
    assert result.panorama.shape == result.classic.shape
    assert (result.panorama[..., 3] > 0).mean() > 0.5
    assert result.metrics["neural_iterations"] >= fast_config.neural.min_iters
    assert any(e.stage == "neural" and e.preview is not None for e in events)
    ref = check.report.reference
    for k, Hk in result.homographies.items():
        assert corner_error(Hk, np.linalg.inv(truth[ref]) @ truth[k], 320, 240) < 1.0


def test_stitch_can_be_cancelled(scene, fast_config, checked):
    images, _ = scene
    with pytest.raises(Cancelled):
        Stitcher(fast_config).stitch(images, checked[1], should_cancel=lambda: True)


def test_without_neural_returns_classic(scene, fast_config, checked):
    images, _ = scene
    fast_config.neural.enabled = False
    result = Stitcher(fast_config).stitch(images, checked[1])
    assert result.panorama is result.classic
    assert "neural_iterations" not in result.metrics


def test_fast_preset_uses_fewer_keypoints(scene):
    from neural_stitch.config import StitchConfig, match_preset
    from neural_stitch.features import extract_features, make_backend
    images, _ = scene
    cfg = StitchConfig(device="cpu")
    cfg.match.backend = "sift"
    fast = match_preset(cfg.match, "fast")
    assert (fast.max_side, fast.num_features) == (768, 1024)
    be = make_backend(cfg.match, None)
    n_acc = len(extract_features(images[0], be, cfg.match).keypoints)
    few = match_preset(cfg.match, "fast").model_copy(update={"num_features": 50})
    n_fast = len(extract_features(images[0], be, few).keypoints)
    assert n_fast <= 50 < n_acc
