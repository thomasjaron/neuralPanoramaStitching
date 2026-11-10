import numpy as np

from neural_stitch.config import NeuralConfig
from neural_stitch.neural import ConvergenceMonitor


def run(losses, motions, cfg):
    m = ConvergenceMonitor(cfg)
    for k, (loss, motion) in enumerate(zip(losses, motions, strict=True), start=1):
        _, stop = m.update(k * cfg.check_every, loss, motion)
        if stop:
            return k * cfg.check_every, m
    return None, m


def test_stops_on_a_flat_loss_once_homographies_are_still():
    cfg = NeuralConfig(min_iters=0, patience=3)
    losses = [1.0 * 0.9**k for k in range(10)] + [0.35] * 10
    stop, m = run(losses, [0.01] * 20, cfg)
    assert stop is not None and stop <= 1600
    assert m.lr_scale < 1


def test_does_not_stop_while_homographies_still_move():
    cfg = NeuralConfig(min_iters=0)
    stop, _ = run([0.3] * 20, [1.0] * 20, cfg)
    assert stop is None


def test_does_not_stop_while_loss_keeps_falling():
    cfg = NeuralConfig(min_iters=0)
    stop, m = run([0.98**k for k in range(30)], [0.0] * 30, cfg)  # 2% per window
    assert stop is None and m.lr_scale == 1


def test_one_good_window_does_not_reset_progress():
    cfg = NeuralConfig(min_iters=0, patience=3, trend_windows=2)
    # flat, flat, a 1% improvement, then flat again
    stop, _ = run([1.0, 1.0, 1.0, 0.99, 0.99, 0.99, 0.99], [0.0] * 7, cfg)
    # plateau counts: -, 1, 2, (good window) 1, 2, 3 -> stop. The old rule (reset to 0) stops at 700.
    assert stop == 600


def test_noise_does_not_hide_a_plateau():
    cfg = NeuralConfig(min_iters=0, patience=3)
    rng = np.random.default_rng(0)
    losses = list(0.5 * (1 + rng.normal(0, 0.004, 40)))  # flat with ±0.4% noise
    stop, _ = run(losses, [0.0] * 40, cfg)
    assert stop is not None and stop <= 1000
