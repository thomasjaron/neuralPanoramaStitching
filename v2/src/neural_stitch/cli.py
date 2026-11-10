"""Command line: ``neural-stitch check|run|eval-sidar|serve``."""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import RESOLUTIONS, StitchConfig, match_preset
from .images import IMAGE_SUFFIXES, InputImage, decode_image, load_folder, save_png
from .pipeline import ProgressEvent, Stitcher


def _load(inputs: list[str], max_side: int) -> list[InputImage]:
    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        return load_folder(inputs[0], max_side)
    return [decode_image(Path(p), Path(p).stem, max_side) for p in inputs
            if Path(p).suffix.lower() in IMAGE_SUFFIXES]


def _config(args: argparse.Namespace) -> StitchConfig:
    cfg = StitchConfig.model_validate_json(Path(args.config).read_text()) if args.config else StitchConfig()
    if args.device:
        cfg.device = args.device
    if args.sift:
        cfg.match.backend = "sift"
    if args.fast:
        cfg.match = match_preset(cfg.match, "fast")
    if args.resolution:
        cfg.output.resolution = args.resolution
    if args.max_side:
        cfg.output.max_side = args.max_side
    if getattr(args, "no_neural", False):
        cfg.neural.enabled = False
    if getattr(args, "max_iters", None):
        cfg.neural.max_iters = args.max_iters
        cfg.neural.min_iters = min(cfg.neural.min_iters, args.max_iters)
    return cfg


def _print_progress(e: ProgressEvent) -> None:
    if e.stage in ("features", "matching"):
        return
    msg = e.message
    if e.stage == "neural" and e.data:
        d = e.data
        rel = d.get("rel_improvement")
        msg += (f" | improvement {rel:.2%}" if rel is not None else "") + \
            f" | moved {d['corner_motion_px']:.3f} px | plateaus {d['plateaus']} | lr x{d['lr_scale']:g}"
    print(f"[{e.stage}] {msg}", flush=True)


def _print_report(check) -> None:
    r = check.report
    print(r.message)
    for im in r.images:
        tag = "reference" if im.reference else ("used" if im.used else f"left out: {im.reason}")
        print(f"  {im.name:>20}  {tag}")
    for key in ("keypoint_rms_ransac_px", "keypoint_rms_bundle_px"):
        if key in check.metrics:
            print(f"  {key}: {check.metrics[key]:.3f}")


def _save(result, images: list[InputImage], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_png(result.panorama, out / "panorama.png")
    save_png(result.classic, out / "panorama_classic.png")
    homs = {images[k].name: H.tolist() for k, H in result.homographies.items()}
    metrics = {k: v for k, v in result.metrics.items() if k != "neural_history"}
    (out / "homographies.json").write_text(json.dumps(homs, indent=2))
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))
    print(f"Saved to {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="neural-stitch", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="JSON file with StitchConfig settings")
        p.add_argument("--device", choices=["auto", "cpu", "cuda"])
        p.add_argument("--sift", action="store_true", help="use SIFT instead of DISK + LightGlue")
        p.add_argument("--fast", action="store_true",
                       help="faster alignment check: smaller matching images, fewer keypoints")
        p.add_argument("--resolution", choices=["native", *RESOLUTIONS],
                       help="output size: a video-style preset (short side, e.g. 1080p) or native")
        p.add_argument("--max-side", type=int, help="long-side limit in pixels for native (default 4000)")

    p = sub.add_parser("check", help="check whether images can be aligned")
    p.add_argument("inputs", nargs="+", help="a folder or image files")
    common(p)

    p = sub.add_parser("run", help="stitch images into a panorama")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--out", default="output")
    p.add_argument("--no-neural", action="store_true", help="skip neural refinement")
    p.add_argument("--max-iters", type=int)
    common(p)

    p = sub.add_parser("eval-sidar", help="stitch a SIDAR folder and compare with ground truth")
    p.add_argument("data_dir", help="folder with rgb/ and homography/")
    p.add_argument("--include-gt", action="store_true",
                   help="also use image 0 (the full ground-truth scene) as an input; makes the task much easier")
    p.add_argument("-o", "--out", default="output")
    p.add_argument("--no-neural", action="store_true")
    p.add_argument("--max-iters", type=int)
    common(p)

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--jobs-dir", help="where uploads and results are kept")
    common(p)

    args = ap.parse_args(argv)
    cfg = _config(args)

    if args.cmd == "serve":
        import uvicorn

        from .app.server import create_app
        uvicorn.run(create_app(Stitcher(cfg), Path(args.jobs_dir) if args.jobs_dir else None),
                    host=args.host, port=args.port)
        return

    stitcher = Stitcher(cfg)
    if args.cmd == "eval-sidar":
        from .evaluate import GT_NAME, corner_errors, load_sidar_truth, pairwise_error
        data = Path(args.data_dir)
        images = load_folder(data / "rgb", cfg.max_input_side)
        if not args.include_gt:
            images = [im for im in images if im.name != GT_NAME]
        print("Inputs:", ", ".join(im.name for im in images))
        result = stitcher.stitch(images, on_progress=_print_progress)
        _print_report(result.check)
        truth = load_sidar_truth(data, [im.name for im in images])
        anchor = result.check.report.reference
        table = {stage: corner_errors(H, images, truth, anchor) for stage, H in [
            ("ransac", result.check.initial), ("bundle", result.check.homographies),
            ("final", result.homographies)]}
        print(f"\nMean corner error vs SIDAR (image-0 pixels, relative to reference {images[anchor].name}):")
        print(f"  {'image':>8}" + "".join(f"{s:>10}" for s in table))
        for name in table["final"]:
            print(f"  {name:>8}" + "".join(f"{table[s][name]:>10.3f}" for s in table))
        print(f"  {'mean':>8}" + "".join(f"{np.mean(list(t.values())):>10.3f}" for t in table.values()))
        stages = [result.check.initial, result.check.homographies, result.homographies]
        pairwise = [pairwise_error(H, images, truth) for H in stages]
        print(f"  {'all pairs':>8}" + "".join(f"{v:>10.3f}" for v in pairwise))
        out = Path(args.out)
        _save(result, images, out)
        (out / "sidar_errors.json").write_text(json.dumps(table, indent=2))
        return

    images = _load(args.inputs, cfg.max_input_side)
    if args.cmd == "check":
        _print_report(stitcher.check(images, on_progress=_print_progress))
        return
    result = stitcher.stitch(images, on_progress=_print_progress)
    _print_report(result.check)
    _save(result, images, Path(args.out))


if __name__ == "__main__":
    main()
