# neural-stitch v2

Panorama stitching for images in **any order**:

1. **Check:** do the images overlap at all?
2. **Rough alignment:** RANSAC.
3. **Least-squares refinement.**
4. **Neural refinement:** runs until the error stops improving.

It comes as a Python library, a command-line tool and a small web app.

```
upload ─► keypoints ─► match all pairs ─► match graph ─► start poses ─► least squares ─► neural refinement ─► panorama
          (DISK, once   (LightGlue +      (which images   (RANSAC chain   (all matches,     (until the loss and
           per image)    RANSAC/MAGSAC)    belong          along the       reference         homographies stop
                                           together?)      strongest       fixed)            changing)
                                                           matches)
```

## Setup

You need [uv](https://docs.astral.sh/uv/). In the `v2/` folder, run:

```bash
uv sync                    # creates .venv and installs everything (dev tools included)
```

Where torch comes from:
- **Linux:** the PyPI build. It includes CUDA and also runs on CPU.
- **Windows:** the CUDA 12.6 index, configured in `pyproject.toml`.

`[tool.uv] exclude-newer` keeps package versions to ones that are at least about two weeks old. Move the date forward to update everything, then run `uv lock --upgrade`.

The DISK and LightGlue weights (about 50 MB) are downloaded on first use into `~/.cache/torch`.

## Usage

```bash
# Web app: http://127.0.0.1:8000
uv run neural-stitch serve

# Command line
uv run neural-stitch check path/to/images/          # can they be aligned?
uv run neural-stitch run path/to/images/ -o out/    # panorama.png, panorama_classic.png, homographies.json, metrics.json
uv run neural-stitch run a.jpg b.jpg c.jpg --no-neural          # RANSAC + least squares only (seconds)
uv run neural-stitch eval-sidar ../data_example -o out/         # compare every stage with SIDAR ground truth

# Options for all commands: --device cpu|cuda, --sift (no downloads), --config settings.json,
#                           --resolution 1080p, --max-side 2000,
#                           --fast (quicker alignment check: 768 px matching images, 1024 keypoints)
```

**Check speed** (5 photo-sized images, CPU, measured 2026-11-10):

| | Total | Keypoints | Matching | Least squares |
|---|---|---|---|---|
| Accurate (default) | 18.4 s | 9.7 s | 6.9 s | 1.7 s |
| Fast (`--fast`) | 8.2 s | 5.8 s | 1.7 s | 0.6 s |

On that test, both settings were about equally accurate. Those test images were enlarged copies of small images, though, so real photos with fine detail may lose a little accuracy in fast mode. Matching time grows with the number of image pairs: 10 images means 45 pairs. In the web app, choose "Matching: Fast / Accurate" before uploading.

**Output resolution.** It works like video resolutions:

- **Presets:** `144p`, `240p`, `360p`, `480p` (SD), `720p` (HD), `1080p` (Full HD), `1440p` (QHD), `2160p` (4K) and `4320p` (8K). The number is the **height**, or the short side for portrait panoramas. The width follows the panorama's shape, so a wide 1080p panorama might be 4200 × 1080.
- **`native`** (the default) uses the images' own resolution, limited to `--max-side` pixels on the long side (default 4000).
- **No upscaling:** a preset larger than the images allow falls back to native size, and the result notes this.
- **Memory limit:** every mode is capped at `max_megapixels` (default 50).
- **Large photos:** uploads over `max_input_side` (3000 px) are shrunk on load. To get larger panoramas from big photos, raise that setting as well.

In the web app, the "Output" dropdown lists every preset with its actual pixel size for your images. Presets above the native resolution are greyed out.

You can change all settings with a JSON file. For the fields, defaults and comments, see [config.py](src/neural_stitch/config.py). Example: `{"neural": {"max_iters": 3000}, "output": {"max_side": 2000}}`.

As a library:

```python
from neural_stitch.images import load_folder
from neural_stitch.pipeline import Stitcher

stitcher = Stitcher()                      # loads the models once; reuse it
images = load_folder("photos/", max_side=3000)
check = stitcher.check(images)             # check.report says which images are used and why others are not
result = stitcher.stitch(images, check)    # result.panorama / result.classic (RGBA), result.homographies
```

## Web app

1. **Upload:** drop in two or more images.
2. **Alignment check:** every image gets a badge: *Reference*, *Used* or *Left out*, with the reason. You also get a quick classic preview and per-pair details.
3. **Neural refinement:** shows live progress (step, loss, improvement, how far the homographies still move), a loss chart and a preview. It can be cancelled.
4. **Result:** the neural render and the classic blend, plus downloads (PNG and homographies as JSON) and timings and errors per stage.

Behind it:
- A FastAPI backend with live progress via Server-Sent Events.
- Two worker threads: one for alignment checks and one for stitching. A new check never waits behind a running refinement, and only one refinement runs at a time.
- Jobs are stored in a temporary folder and deleted after 6 hours.

| Endpoint | |
|---|---|
| `POST /api/jobs` | Upload images (multipart `files`); the check starts automatically |
| `GET /api/jobs/{id}` | State, report, metrics, files |
| `POST /api/jobs/{id}/start` | `{"neural": true, "max_iters": 6000}` |
| `POST /api/jobs/{id}/cancel`, `DELETE /api/jobs/{id}` | |
| `GET /api/jobs/{id}/events` | Live progress (SSE) |
| `GET /api/jobs/{id}/files/{name}` | Thumbnails, previews, results (`?download=true`) |

## How each step works

- **Match graph** ([graph.py](src/neural_stitch/graph.py)):
  - Two images are linked when RANSAC finds enough consistent matches and the warp is plausible: not mirrored, folded, sent to infinity, or wildly resized.
  - The largest connected group is kept.
  - The best-connected image becomes the reference.
  - Start poses are chained along the strongest links (a maximum spanning tree).
- **Least squares** ([bundle.py](src/neural_stitch/bundle.py)):
  - All homographies are refined together on all inlier matches, using LBFGS and a robust Cauchy loss.
  - The error is measured in image pixels in both directions, so the solver can't cheat by shrinking images.
- **Neural refinement** ([neural.py](src/neural_stitch/neural.py)):
  - An MLP with coarse-to-fine positional encoding (BARF-style) learns the panorama colours.
  - At the same time, it learns small homography corrections (sl(3) parameters) and a per-image exposure correction.
  - Each step uses random patches from every image. The loss combines colour, colour gradients and keypoints.
- **Convergence** (`ConvergenceMonitor` in [neural.py](src/neural_stitch/neural.py)):
  - Every `check_every` steps, the mean loss of that window is recorded.
  - The improvement is the **trend** over the last `trend_windows` windows (the slope of log(loss)), so one noisy window can't fake or hide a plateau.
  - A trend below `rel_tol` counts as a plateau and halves the learning rates. A window that still improves lowers the plateau count by one; it doesn't reset it.
  - Training stops after `patience` plateaus, once the homographies also move less than `corner_tol` pixels.
  - `max_iters` is a safety cap.

## Results on `data_example` (SIDAR)

`eval-sidar` leaves out image `0` by default, because it's the complete ground-truth scene. As an input, it would hide misalignments in the blend and act as an anchor for all other images. (`--include-gt` puts it back.) Errors are measured relative to the stitched reference image and given in image-0 pixels.

Measured on 2026-10-23 (CPU), with 5 inputs (`2, 5, 15, 17, 48`):

| Stage | vs reference image (px) | all pairs (px) |
|---|---|---|
| RANSAC chain | 0.235 | 0.361 |
| + least squares | 0.286 | 0.418 |
| + neural refinement (stopped on its own at step 4,100, 15 min) | 0.314 | 0.417 |

All stages are already well below one pixel. The differences between them (about 0.05 px) are smaller than the spread between single images, so this one small scene can't rank the methods. For a reliable comparison, evaluate several SIDAR scenes, ideally larger images with less overlap.

Timings on CPU (AMD laptop):

| Step | Time |
|---|---|
| Keypoints | 2.7 s |
| Matching (15 pairs) | 2.8 s |
| Least squares | 1.8 s |
| Neural refinement | about 0.28 s per step |

A CUDA GPU should make the neural step much faster.

## Tests

```bash
uv run pytest       # synthetic scenes with known homographies; uses SIFT, so no downloads; about 5 s
uv run ruff check src tests
```

## Known limits

- **The neural render is softer than the classic blend** at the default model size and step counts. Both are always saved. Sharper results need longer training or a hash-grid encoding.
- **Homographies only:** this assumes a rotating camera or a flat scene. A strong parallax shows up as ghosting.
- Jobs live in memory, so restarting the server forgets them.
- HEIC photos aren't supported (Pillow needs a plugin).
