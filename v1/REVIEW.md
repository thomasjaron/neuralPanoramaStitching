# Code Review: neuralPanoramaStitching (v1)

> Status 2026-10-23: the thesis code was moved to `v1/` and stays unchanged for reproducibility. The rebuild in [`../v2`](../v2) covers the target picture from sections 6 and 7: match graph, RANSAC start, least squares, neural refinement until convergence, and the web app.

_Date: 2026-10-23 · Scope: all of `src/`, config, build files_

> **Note:** The code was **not run**, because this machine has no Python environment with torch. Every bug below comes from reading the code. Each fix should come with a small test that confirms the bug first.

## TL;DR

- **Does it work?** Yes, on the example data. Several settings crash, though, and a few quiet bugs affect results or metrics. One example: the `set_estimated_homs` switch has no effect, and the RANSAC baseline metric is broken (B1).
- **Is there a more efficient way?** Yes. The three biggest wins:
  1. Compute image features **once per image**, not once per pair (E1).
  2. For the frontend, add a **fast mode**: start from RANSAC and line up the keypoints with a direct least-squares solve instead of 5,000 training steps. The from-scratch research mode stays as it is (E3).
  3. Drop Lightning and use a plain training loop. Right now the images are re-read from disk on every step (E4).
- **Frontend-ready?** Not yet. The core has to become a clean function that takes images and returns a report or a panorama: any image order, any file names, any image sizes. See sections 6 and 7.

---

## 1. What the code does today

1. Loads the images listed in `options.yaml` from `rgb/`.
2. For every pair of images: finds keypoints (DISK), matches them (LightGlue), and estimates a homography (OpenCV MAGSAC).
3. Drops images that don't connect to the **first** image.
4. Trains for `max_iter` steps (10,000):
   - **Phase A** (first 50%): only the homographies move, so that matched keypoints line up.
   - **Phase B** (second 50%): a small neural network (the "neural image") also learns to paint the panorama, using a color loss and a gradient/blending loss.
5. Saves preview PNGs to `output/vis`.

The idea is solid and the code is small (~1,400 lines). It's still a research script, though: hard-coded paths, no tests, global settings that get changed at runtime, and the ground-truth (SIDAR) evaluation mixed into the main code.

---

## 2. Bugs

Sorted by impact.

| # | Where | Problem | Effect |
|---|---|---|---|
| B1 | [datasets.py:180-182](src/datasets.py#L180-L182) | Starting from scratch is the thesis setup (how precise is the model without help?), and pinning image 0 to its SIDAR pose is intended. However, `homographies[0] = gt_homographies[0]` always overwrites the starting pose of **every** image, so the `set_estimated_homs` switch ([model.py:41-43](src/model.py#L41-L43)) does nothing. The `RANSAC_TO_SIDAR` metric reads the same overwritten row. Fix: if `set_estimated_homs` is on, keep RANSAC (with SIDAR: `gt[0] @ row`); otherwise keep today's from-scratch start. | The example config says `set_estimated_homs: true` but actually runs from scratch, which is misleading. The RANSAC-vs-SIDAR baseline compares ground truth with ground truth, so the comparison with classic RANSAC is missing. |
| B2 | [datasets.py:220](src/datasets.py#L220), [228](src/datasets.py#L228) | Wrong multiplication order when a missing homography is built through a third image. `cv2.findHomography(pts2, pts1)` makes `H[i][j]` map image j → image i, so the chain must be `H[i][k] @ H[k][j]`. The code computes `H[k][j] @ H[i][k]`. | Images that connect only through another image get a wrong starting position. |
| B3 | [datasets.py:256-270](src/datasets.py#L256-L270) | After the unconnected images are removed, `selected_images` keeps its old length but is indexed with the new positions. | When any image is dropped, the wrong image can be removed, and the image list no longer matches the homographies. |
| B4 | [datasets.py:151-154](src/datasets.py#L151-L154) | The `estimate_homs: false` path hard-codes `repeat(5, 5, …)`. It also never sets `images`, `keypoints` or `used_idxs`. | Crashes (`UnboundLocalError`). |
| B5 | [model.py:64](src/model.py#L64), [model.py:200](src/model.py#L200), [networks.py:214](src/networks.py#L214) | Phase B only switches on correctly when `loss_weights[2]` is **exactly 1**. | Any other value either crashes at step 0 (`pred.rgb_map` is missing) or leaves the neural image's learning rate at 0 forever. |
| B6 | [model.py:229](src/model.py#L229), [networks.py:220](src/networks.py#L220) | `if tb and A or B` is evaluated as `(tb and A) or B`. | With `tb: false`, the last step calls a logger that is `None`, so the run crashes at the very end. |
| B7 | [networks.py:249](src/networks.py#L249) | `self.barf_c2f` is only set when `posenc: true`. | `posenc: false` crashes. |
| B8 | [model.py:211-216](src/model.py#L211-L216) | "Fix the first image" resets its weights **before** the optimizer step, so the step moves the image again. In SIDAR mode it is reset to the ground-truth pose. | The reference image drifts a little on every step. Training also uses ground truth, which is fine for experiments but not for a product. |
| B9 | [config.py:36](src/config.py#L36) | No defaults for `loss_weights`, `loss_threshold`, `blending_depth`, `blending_resize_factor`, `log_image`, `log_scalar`, `output_H/W` or `set_estimated_homs`. No type or value checks. | A shorter `options.yaml` crashes deep inside training with an unclear error. |
| B10 | [\_\_init\_\_.py:2](src/__init__.py#L2), [model.py:14](src/model.py#L14), [model.py:293](src/model.py#L293) | `__init__.py` imports `src.model`, but the modules use `from config import …`. The data path `./data_example` is hard-coded. | Only `python src/model.py`, run from the repo root, works. `import src` fails. |
| B11 | [datasets.py:44](src/datasets.py#L44), [datasets.py:175](src/datasets.py#L175) | Images are looked up with `startswith(f"{name}.")` and stacked into one tensor. | File names must look like `<name>.png`, and all images must be the same size. This blocks "upload any images". |
| B12 | [model.py:92-113](src/model.py#L92-L113) | Only 20 box colors are defined. | `vis_hom: true` with more than 20 images raises an `IndexError`. |

**Smaller issues:**
- Unused imports: `AdalamFilter` and `matplotlib` ([estimators.py:3-5](src/estimators.py#L3-L5)).
- The `disk-epipolar` option loads a model and then raises an error right away ([estimators.py:32-34](src/estimators.py#L32-L34)).
- `torch.meshgrid` is called without `indexing=`, which prints a warning ([utils.py:168](src/utils.py#L168)).
- `imageio.imsave` is deprecated.
- `colorcode_to_number` re-implements `PIL.ImageColor.getrgb`.
- `_scale_homographies` divides by `det = 0` for missing pairs, which produces NaN, and it indexes the unfiltered image list.
- The README says `src/models.py`; the file is `src/model.py`.

---

## 3. Efficiency

| # | Today | Better | Gain |
|---|---|---|---|
| E1 | DISK features are recomputed for **every pair** ([datasets.py:279](src/datasets.py#L279)), so each image is processed N-1 times. | Compute features once per image, keep them, and only run the matching per pair. | N× fewer DISK runs. For 10 images that's 10 runs instead of 90. |
| E2 | Matching runs on full-size images. | Shrink images to about 1024 px on the long side for matching, then scale the keypoints back up. | Large gain for phone photos (12 MP+). Also avoids GPU out-of-memory errors. |
| E3 | Phase A spends 5,000 Adam steps lining up keypoints from scratch. This is the thesis experiment, so keep it as the **research mode**. | For the frontend, add a **fast mode**: start from RANSAC (needs the B1 fix), then solve the keypoint alignment directly with least squares, for example Levenberg–Marquardt via `scipy.optimize.least_squares` or torch LBFGS ("bundle adjustment"). | In fast mode, half the training time shrinks to about a second. It's also a useful extra baseline for comparing against the research mode. |
| E4 | Lightning runs with `max_steps` on a one-batch dataset, which makes 10,000 "epochs". The DataLoader re-reads every image from disk on each step, but the images are only used once ([model.py:196-197](src/model.py#L196-L197)). The model also rebuilds its networks and patches the optimizer inside `training_step` to work around Lightning. | A plain PyTorch loop: load the images to the GPU once and build the model once. | Less overhead and much simpler code. It also makes it easy to send progress to a frontend. |
| E5 | Phase B sends **every pixel of every image** through the MLP on every step. | Sample random pixels or patches per step, as NeRF does. Use mixed precision (bf16). Optionally replace the plain MLP with a hash-grid encoding (Instant-NGP style). | Potentially very large. Check that quality holds (PSNR/SSIM on SIDAR). |
| E6 | The gradient/pyramid loss loops over images in Python, and the Sobel/Gaussian kernels and the pixel grid are rebuilt every step. `forward()` also warps the grid during Phase A, where the result isn't used ([networks.py:50](src/networks.py#L50)). | Process all images in one batched call. Create the kernels and grids once. Skip the warp during Phase A. | Moderate. |
| E7 | Always runs the full `max_iter`. | Stop early once the loss stops improving. | Often large. |
| E8 | Every pair of images is matched (N²). | Above about 20 images, only match likely neighbors, found first with global image descriptors on thumbnails. | Only matters for large image sets. |

The "can these images align?" check needs **no training at all**. E1, E2 and the match graph from section 6 answer it in seconds.

---

## 4. Structure and maintainability

- **Settings change at runtime.** `H`, `W`, `output_H/W` and `batch_size` are overwritten during setup ([datasets.py:169-174](src/datasets.py#L169-L174), [model.py:246-251](src/model.py#L246-L251)). This is hard to follow and a source of bugs. Make the config read-only and pass sizes explicitly.
- **`rescale_factor` does two jobs.** It shrinks the training images **and** sets the canvas size (with 0.2, the canvas is 5× the reference image). Images that land outside this fixed canvas are cut off; with only a few images, most of the canvas stays empty. The canvas should be computed from the warped image corners instead.
- **SIDAR/ground-truth code is mixed into the main pipeline** (`use_sidar` branches everywhere). Move it into a separate `evaluation` module.
- **Results only exist as PNG files** in `output/vis`. A function should return the image and the homographies.
- No tests, no CI, and `print` instead of logging.
- No LICENSE file. The code is adapted from BARF (MIT license), so keep its notice.

---

## 5. Dependencies and tooling

**Problems:**
- **The CUDA install breaks the pinned torch version.** `requirements.txt` pins `torch==2.7.1`, but `make install-cuda-deps` installs from the `cu124` index, whose newest torch is **2.6.0** (I checked the index). Torch gets downgraded, and the lock file no longer matches what's installed.
- `make install-deps` re-runs `pip-compile` on every install, so the lock file is regenerated each time and installs aren't reproducible. The current lock was made on Windows (it contains `colorama`).
- **Used but not listed directly:** `torchvision`, `pyyaml`, `pillow`, `torchmetrics`.
- **Listed but unused:** `scikit-video` (abandoned since 2018) and `matplotlib` (imported, never used).
- `easydict` is unmaintained. Replace it with a dataclass or a pydantic model.
- `.pylintrc` is saved as **UTF-16 with Windows line endings**, which most tools can't read reliably.
- **Dev tools:** `bumpversion` and `bump2version` are both abandoned. `autopep8`, `pycodestyle` and `pylint` overlap, and **ruff** replaces all three and is much faster. `types-setuptools` isn't needed. `pytest` is installed, but there are no tests.
- On a server, use `opencv-python-headless` instead of `opencv-python`.

**Versions** (latest on PyPI as of 2026-10-23):

| Package | Now | Latest | Note |
|---|---|---|---|
| Python | 3.12.6 tested | – | Stay on 3.12 or newer. numpy 2.5 and scipy 1.18 require it. |
| torch / torchvision | 2.7.1 / not listed | 2.14.1 / 0.29.1 | Use the `cu126` or `cu130` index. `cu124` stops at 2.6. |
| lightning | 2.5.1 | 2.6.6 | Or remove it (E4). |
| kornia | 0.8.1 | 0.8.3 | |
| numpy | 2.2.6 | 2.5.4 | |
| scipy | 1.15.3 | 1.18.1 | |
| opencv-python | 4.11 | 5.0.0.93 | **Major version.** Test `findHomography` / `USAC_MAGSAC` before switching. |
| tensorboard | 2.19 | 2.21 | |
| torchmetrics | 1.7.2 (indirect) | 1.9.0 | Add it as a direct dependency. |
| imageio | 2.37 | 2.38.1 | Switch to `imageio.v3`. |
| matplotlib | 3.10.3 | 3.11.2 | Unused, remove. |
| scikit-video | 1.1.11 | 1.1.11 (2018) | Remove. |
| easydict | 1.13 | 1.13 (2024) | Replace. |
| pylint / mypy / pytest | 3.3.7 / 1.16 / 8.4 | 4.1.3 / 2.4.0 / 9.1.1 | Replace pylint with ruff; keep mypy and pytest. |

**Recommendation:** move to `pyproject.toml` with **uv**. It gives one lock file that works on Linux and Windows, can point torch at the right CUDA index, and is much faster than pip-tools. The Makefile shrinks to a few lines. Only pin versions that have been out for at least two weeks.

---

## 6. The target: a frontend for "any images, any order"

```
Upload ──► Prepare ──► Check alignment ──► Initial alignment ──► Neural refine ──► Result
any order   EXIF rotate  features once      pick best-connected   GPU job,          PNG +
any names   to RGB       match pairs        image as reference,   progress +        homographies
any size    shrink       build match graph  chain + least squares live preview,     (JSON)
                         → which images fit? quick classic preview early stop
```

- **Check alignment** (takes seconds):
  - Each image is a node in a "match graph". Two images get an edge when they have a good match: enough inliers and a sane homography (positive determinant, not too skewed, corners not flipped).
  - Keep the largest connected group of images, and tell the user which images were left out and why.
  - Stop if fewer than two images fit.
  - This replaces the current "must connect to the first image" rule, so the upload order no longer matters.
- **Initial alignment:**
  - The reference image is the best-connected one, not the first in the list.
  - Chain the homographies along the strongest matches (a maximum spanning tree), then refine them all at once (E3).
  - A quick OpenCV warp-and-blend gives the user an instant preview.
- **Neural refine:** today's method, run as a background job on a single GPU worker. It reports the step, the loss and a preview image every N steps, and the user can cancel it.
- **Backend:** FastAPI with these endpoints:
  - `POST /jobs` (upload the images)
  - `POST /jobs/{id}/check`
  - `POST /jobs/{id}/start`
  - `GET /jobs/{id}/events` (live progress via SSE)
  - `GET /jobs/{id}/result`

  Also: one GPU worker with a queue, limits on image count and size, cleanup of temporary files, and loading the DISK/LightGlue weights once at startup (today they're downloaded on first use).
- **Frontend:**
  1. Drag-and-drop upload.
  2. Alignment report: thumbnails, with the images in use marked green and the ones left out grey.
  3. Start button.
  4. Progress bar with a live preview.
  5. Download.

  **Gradio** makes a quick one-file prototype. A small React/Vite or Svelte app gives full control later.

---

## 7. Task list

One task = one ticket/PR. Size: **S** < 1 day, **M** = 1–3 days, **L** = more than 3 days.

### Phase 0: Foundation

| ID | Task | Done when | Size | Needs |
|---|---|---|---|---|
| T00 | Tag the thesis version (`git tag thesis-2025-06`). Save its metrics on the example data (SIDAR errors, SSIM, PSNR, runtime) as reference numbers. | Every later change can be checked against these numbers. | S | – |
| T01 | Move to `pyproject.toml` + uv on Python 3.12. Fix the torch CUDA index. Remove unused dependencies and list the direct ones. | A fresh clone runs after `uv sync` on Linux and Windows. | S–M | – |
| T02 | Update dependencies to the versions in section 5. Do OpenCV 5 last, as its own step. | The example runs and the output looks the same as before. | S | T01 |
| T03 | Replace autopep8, pycodestyle, pylint and bumpversion with ruff (plus mypy). Delete the UTF-16 `.pylintrc`. Add a GitHub Actions workflow that runs lint and tests. | CI passes on push. | S | T01 |
| T04 | Turn `src` into a real package (e.g. `neural_stitch/`) with consistent imports, and add a CLI: `neural-stitch run <data_dir> [--iters N]`. | Runs from any folder. | M | T01 |
| T05 | Tests: sl3↔SL3 round trip, homography chaining, image filtering, plus a smoke test (CPU, 50 steps, example data). | `pytest` passes in CI. | M | T04 |
| T06 | Benchmark script that measures time and memory per stage (matching, Phase A, Phase B). | Baseline numbers are written down. | S | T04 |

### Phase 1: Bug fixes (each with a test)

| ID | Task | Size | Needs |
|---|---|---|---|
| T07 | Fix B1: make `set_estimated_homs` work. `false` keeps today's from-scratch start (the research mode, unchanged). `true` keeps the RANSAC guess; with SIDAR, apply `gt[0] @ row`. Fix the `RANSAC_TO_SIDAR` metric so it reads the real RANSAC result. Set the example config to the value the thesis actually used. Test: with `false`, results match T00. With `true`, image 0 starts at `gt[0]` and the other images start at `gt[0] @ H_ransac`. | S | T05, T00 |
| T08 | Fix B2: correct the multiplication order in the homography chain. | S | T05 |
| T09 | Fix B3: index mismatch when images are dropped. | S | T05 |
| T10 | Fix B4: make `estimate_homs: false` work, or remove the option. | S | – |
| T11 | Fix B5: add an explicit setting for when Phase B starts, instead of checking `loss_weights[2] == 1`. | S | – |
| T12 | Fix B6, B7 and B12 (small crashes) and the smaller issues from section 2. | S | – |
| T13 | Fix B8: truly fix the reference image (zero its gradient or don't make it a parameter). Use ground truth only in evaluation mode. | S | – |
| T14 | Fix B9: config as a dataclass or pydantic model with every default and validation. Remove easydict and stop changing the config at runtime. | M | T04 |

### Phase 2: Speed (measure with T06 before and after)

| ID | Task | Size | Needs |
|---|---|---|---|
| T15 | Features once per image + matching on smaller images (E1, E2). | M | T04 |
| T16 | Replace Lightning with a plain training loop: images go to the GPU once, and a progress callback reports each step (E4). | M | T04 |
| T17 | Add a fast mode: RANSAC start + least-squares alignment instead of Phase A (E3). The research mode stays the default for the thesis setup. | M | T07, T08 |
| T18 | Faster Phase B: pixel sampling, batched loss, cached kernels and grids, bf16 (E5, E6). Compare quality against the current version. | L | T16 |
| T19 | Early stopping (E7). | S | T16 |
| T20 | *(Optional)* Experiment with a hash-grid encoding in place of the MLP. | L | T18 |

### Phase 3: Core for "any images, any order"

| ID | Task | Size | Needs |
|---|---|---|---|
| T21 | Input handling: any file names, sizes and formats; EXIF rotation; RGBA → RGB; a maximum image size. | S–M | T04 |
| T22 | Match graph + alignment check that returns a report (which images are used, which are left out, and why), including homography sanity checks. | M | T15 |
| T23 | Pick the best-connected image as reference and chain homographies along a spanning tree. This replaces `create_startvalue_homographies`. | M | T22, T17 |
| T24 | Compute the canvas from the warped corners. Separate the training scale from the output scale. Render the full-size result. | M | T16 |
| T25 | Public API: `check(images) -> Report` and `stitch(images, cfg, on_progress) -> Result`, with no file I/O inside. Move SIDAR into `evaluation/`. | M | T16, T22–T24 |
| T26 | Quick classic preview (OpenCV warp + multi-band blend) from the initial homographies. | S | T23 |

### Phase 4: Web app

| ID | Task | Size | Needs |
|---|---|---|---|
| T27 | FastAPI backend with the job endpoints and SSE progress. | M | T25 |
| T28 | Job runner: one GPU worker with a queue, cancel, timeouts, temp-file cleanup and upload limits. | M | T27 |
| T29 | Frontend: upload → report → progress/preview → download. Start with a Gradio prototype; build a React/Svelte app later if needed. | M–L | T27 |
| T30 | Docker image (CUDA base, model weights included) + deploy notes. | S–M | T27 |
| T31 | Update the README: install, CLI, API and frontend. | S | T29 |

**Suggested order:**
1. T01–T06.
2. T07–T14: quick wins in result quality.
3. T15–T17: the biggest speed wins.
4. Phase 3.
5. Phase 4.

T18–T20 can run alongside Phase 4.
