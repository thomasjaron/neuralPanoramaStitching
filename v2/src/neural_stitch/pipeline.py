"""Public API: ``Stitcher.check(images)`` and ``Stitcher.stitch(images)``.

check:  features (once per image) -> pair matching + RANSAC -> match graph -> start poses
        -> least-squares refinement -> quick classic preview
stitch: check (if not given) -> neural refinement until convergence -> render
"""
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch

from .bundle import refine_homographies
from .config import MatchConfig, OutputConfig, StitchConfig, resolve_device
from .features import Backend, PairMatch, extract_features, make_backend, match_pair
from .graph import AlignmentReport, build_report
from .images import InputImage
from .neural import Cancelled, NeuralImage, refine_neural
from .render import canvas_bounds, native_size, output_transform, render_classic, render_field

__all__ = ["Cancelled", "CheckResult", "ProgressEvent", "StitchResult", "Stitcher"]


@dataclass
class ProgressEvent:
    stage: str  # features | matching | bundle | neural | render
    message: str
    fraction: float | None = None  # 0..1 within the stage, if known
    data: dict = field(default_factory=dict)
    preview: np.ndarray | None = None  # RGBA


ProgressFn = Callable[[ProgressEvent], None]
CancelFn = Callable[[], bool]


@dataclass
class CheckResult:
    report: AlignmentReport
    initial: dict[int, np.ndarray]  # RANSAC chain: image -> reference pixels
    homographies: dict[int, np.ndarray]  # after least squares
    pairs: list[PairMatch]  # accepted pairs inside the main group
    preview: np.ndarray | None  # RGBA, classic blend
    metrics: dict


@dataclass
class StitchResult:
    check: CheckResult
    homographies: dict[int, np.ndarray]  # final: image -> reference pixels
    panorama: np.ndarray  # RGBA; neural render if enabled, else classic
    classic: np.ndarray  # RGBA; classic blend with the final homographies
    metrics: dict


class Stitcher:
    """Holds the feature models, so create it once and reuse it (e.g. in a server)."""

    def __init__(self, config: StitchConfig | None = None):
        self.config = config or StitchConfig()
        self.device = resolve_device(self.config.device)
        self._backend: Backend | None = None

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = make_backend(self.config.match, self.device)
        return self._backend

    def check(self, images: list[InputImage], on_progress: ProgressFn | None = None,
              should_cancel: CancelFn | None = None, match: MatchConfig | None = None) -> CheckResult:
        """``match`` overrides the matching settings for this call (e.g. a speed preset)."""
        emit = on_progress or (lambda e: None)
        cancelled = should_cancel or (lambda: False)
        cfg = self.config
        mcfg = match or cfg.match
        t0 = time.perf_counter()

        feats = []
        for k, img in enumerate(images):
            if cancelled():
                raise Cancelled()
            emit(ProgressEvent("features", f"Finding keypoints in {img.name}", k / len(images)))
            feats.append(extract_features(img, self.backend, mcfg))
        t1 = time.perf_counter()

        combos = list(combinations(range(len(images)), 2))
        pairs = []
        for c, (i, j) in enumerate(combos):
            if cancelled():
                raise Cancelled()
            emit(ProgressEvent("matching", f"Matching {images[i].name} and {images[j].name}",
                               c / max(1, len(combos))))
            pairs.append(match_pair(i, j, feats[i], feats[j], images[j].size, self.backend, mcfg))
        t2 = time.perf_counter()

        report, initial = build_report([im.name for im in images], pairs)
        metrics = {"time_features_s": t1 - t0, "time_matching_s": t2 - t1}
        if not report.ok:
            return CheckResult(report, {}, {}, [], None, metrics)

        used = [p for p in pairs if p.ok and p.i in initial and p.j in initial]
        emit(ProgressEvent("bundle", "Refining all homographies on the keypoints"))
        sizes = {k: images[k].size for k in initial}
        refined, stats = refine_homographies(initial, sizes, used, report.reference, cfg.bundle)
        t3 = time.perf_counter()
        metrics |= {"time_bundle_s": t3 - t2, "keypoint_rms_ransac_px": stats["rms_before_px"],
                    "keypoint_rms_bundle_px": stats["rms_after_px"], "matches": stats["matches"]}

        idxs = sorted(refined)
        rgbs, Hs = [images[k].rgb for k in idxs], [refined[k] for k in idxs]
        metrics["native_size"] = list(native_size(canvas_bounds(Hs, [images[k].size for k in idxs])))
        preview, _ = self._classic(rgbs, Hs, OutputConfig(max_side=1024))
        emit(ProgressEvent("bundle", "Alignment check done", 1.0, preview=preview))
        return CheckResult(report, initial, refined, used, preview, metrics)

    def stitch(self, images: list[InputImage], check: CheckResult | None = None,
               config: StitchConfig | None = None, on_progress: ProgressFn | None = None,
               should_cancel: CancelFn | None = None) -> StitchResult:
        """Run the full pipeline. ``config`` overrides the neural/output settings for this run."""
        cfg = config or self.config
        emit = on_progress or (lambda e: None)
        check = check or self.check(images, on_progress, should_cancel)
        if not check.report.ok:
            raise ValueError(check.report.message)
        ref = check.report.reference
        idxs = sorted(check.homographies)
        sizes = [images[k].size for k in idxs]
        metrics = dict(check.metrics)
        H = check.homographies
        model: NeuralImage | None = None
        to_field = None

        if cfg.neural.enabled:
            t0 = time.perf_counter()
            ncfg = cfg.neural

            def preview(m: NeuralImage, C: np.ndarray, Hs: dict[int, np.ndarray]) -> np.ndarray:
                return self._neural([Hs[k] for k in idxs], sizes, m, C, [images[k].rgb for k in idxs],
                                    OutputConfig(max_side=ncfg.preview_side))[0]

            def on_check(entry: dict, make_preview: Callable[[], np.ndarray] | None) -> None:
                emit(ProgressEvent("neural", f"Step {entry['iter']}: loss {entry['loss']:.5f}",
                                   entry["iter"] / ncfg.max_iters, data=entry,
                                   preview=make_preview() if make_preview else None))

            res = refine_neural({k: images[k].rgb for k in idxs}, H, check.pairs, ref, ncfg, self.device,
                                on_check=on_check, should_cancel=should_cancel, preview=preview,
                                seed=cfg.seed)
            H, model, to_field = res.homographies, res.model, res.to_field
            metrics |= {"time_neural_s": time.perf_counter() - t0, "neural_iterations": res.iterations,
                        "neural_stop_reason": res.stop_reason, "keypoint_rms_neural_px": res.rms_px,
                        "neural_history": res.history}

        emit(ProgressEvent("render", "Rendering the panorama"))
        t0 = time.perf_counter()
        rgbs = [images[k].rgb for k in idxs]
        Hs = [H[k] for k in idxs]
        classic, note = self._classic(rgbs, Hs, cfg.output)
        panorama = classic
        if model is not None and to_field is not None:
            panorama, _ = self._neural(Hs, sizes, model, to_field, rgbs, cfg.output, alpha=classic[..., 3])
        metrics["time_render_s"] = time.perf_counter() - t0
        metrics["output_resolution"] = cfg.output.resolution
        metrics["output_size"] = f"{classic.shape[1]} × {classic.shape[0]}"
        if note:
            metrics["output_note"] = note
        emit(ProgressEvent("render", "Done", 1.0))
        return StitchResult(check=check, homographies=H, panorama=panorama, classic=classic, metrics=metrics)

    def _classic(self, rgbs: list[np.ndarray], Hs: list[np.ndarray],
                 out: OutputConfig) -> tuple[np.ndarray, str | None]:
        sizes = [(im.shape[1], im.shape[0]) for im in rgbs]
        T, size, note = output_transform(canvas_bounds(Hs, sizes), out)
        return render_classic(rgbs, Hs, T, size), note

    def _neural(self, Hs: list[np.ndarray], sizes: list[tuple[int, int]], model: NeuralImage,
                to_field: np.ndarray, rgbs: list[np.ndarray], out: OutputConfig,
                alpha: np.ndarray | None = None) -> tuple[np.ndarray, str | None]:
        T, size, note = output_transform(canvas_bounds(Hs, sizes), out)
        if alpha is None or alpha.shape[:2] != (size[1], size[0]):
            alpha = render_classic(rgbs, Hs, T, size)[..., 3]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return render_field(model, to_field, T, size, alpha, self.device), note
        finally:
            model.train(was_training)
