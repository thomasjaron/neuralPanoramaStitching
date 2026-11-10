# Neural Panorama Stitching

Code for the master's thesis "Neural Panorama Stitching" (June 2025) by Thomas Jaron-Strugala, plus a rebuilt version with a web app.

| Folder | What |
|---|---|
| [v1/](v1/) | The thesis code. It learns the homographies from scratch, with no RANSAC start, to measure how precise the model is without help. It also has a [code review](v1/REVIEW.md). |
| [v2/](v2/) | The rebuild (uv project). It checks whether images can be aligned, starts from RANSAC, refines with least squares, then does neural refinement until convergence. Includes a CLI and a web app. See [v2/README.md](v2/README.md). |
| [data_example/](data_example/) | SIDAR example scene with ground-truth homographies, used by both versions. |

Quick start for v2:

```bash
cd v2
uv sync
uv run neural-stitch serve      # open http://127.0.0.1:8000
```

The code ideas are inspired by [BARF: Bundle-Adjusting Neural Radiance Fields](https://github.com/chenhsuanlin/bundle-adjusting-NeRF) (Lin et al., ICCV 2021).
