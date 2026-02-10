# PLAN.md — Tennis Virtual Ad Overlay Baseline (Single Broadcast Feed)

## Progress

- [x] **Step 0** — Repo scaffold (`pyproject.toml`, `configs/`, `scripts/run_video.py`, project layout)
- [x] **Step 1** — Video I/O + deterministic frame loop (`VideoReader`, `VideoWriter`, frame-index overlay)
- [x] **Step 2A** — Calibrator interface + dummy wiring (`CourtCalibrator` ABC, `DummyCalibrator`, `--calibrator` flag)
- [x] **Step 2B** — Single-image calibrator proof (TennisCourtDetector wrapper + debug script)
- [x] **Step 2C** — Video integration (calibrator in run_video.py, court overlay per frame)
- [x] **Step 3A** — Temporal keypoint smoothing (EMA smoother + error spike reset)
- [x] **Step 4A** — Ad warp + naive composite (AdPlacer, placement spec, alpha blend)
- [ ] **Step 3B** — Scene-cut detection (reset smoother on camera changes)
- [ ] **Step 4** — Temporal stabilizer for homography (direct H smoothing, if needed)
- [x] **Step 5A** — OcclusionMasker v1 (players only — Mask R-CNN person segmentation + ad occlusion)
- [ ] **Step 5B** — OcclusionMasker v2 (ball / net / shadows)
- [ ] **Step 7** — Composite with occlusion (advanced blending)
- [ ] **Step 8** — Debug views

---

## Goal

Build an end-to-end baseline that:

* takes a tennis broadcast video (or stream frames),
* estimates court homography per frame (stable over time),
* warps an RGBA ad into a chosen court region,
* masks occluders (players first; later ball/net/ballkids/racket),
* composites the result into an output video at ~30 fps target (optimize later).

Primary objective: **modular pipeline** so we can swap calibrators / maskers / blending methods quickly.

---

## Repo leverage strategy (based on your inventory)

### Court calibration backbone (choose 1–2 to start)

* **Primary baseline:** `TennisCourtDetector` (heatmap keypoints + refine + homography correction)

  * Reason: dataset + weights exist; has video inference; designed for broadcast frames; already supports refine + homography best-config.
* **Secondary option:** `CourtCheck/models/court_detection/*`

  * Reason: contains nice homography selection (`homography.py` with 12 configs) + stabilization + scene cut detection ideas.
* **Fallback (debug / sanity):** `tennis_court_detection` (classical CV)

  * Reason: slow, but great to validate geometry and as a "ground truth-ish" sanity check on a few frames.

### Occlusion baseline (not in your repos)

None of your 4 repos provide pixel-accurate occlusion masks (they do bboxes, ball coords). So for MVP:

* use an off-the-shelf **person segmentation** model (fast and easy)
* later optionally fine-tune on tennis.

### Ball / bounce (optional for MVP)

* CourtCheck has TrackNet + bounce detector. Good later, but not required for "player occludes ad" demo.

---

## System architecture (interfaces)

Implement these as small Python modules with stable signatures.

### A) `CourtCalibrator`

**Input:** frame (HWC BGR/RGB)
**Output:** homography `H` (court→image OR image→court, but be consistent), confidence, debug

```python
class CourtCalibrator:
    def estimate(self, frame) -> dict:
        return {
          "H": np.ndarray(shape=(3,3)),
          "conf": float,
          "keypoints": np.ndarray,  # optional
          "debug": dict,            # overlays, reprojection error, etc.
        }
```

### B) `OcclusionMasker`

Returns a mask in image coordinates: 1 = occluder (foreground), 0 = background.

```python
class OcclusionMasker:
    def mask(self, frame) -> dict:
        return {
          "mask": np.ndarray(shape=(H,W), dtype=float),  # allow soft mask
          "conf": float,
          "debug": dict
        }
```

### C) `AdPlacer`

Warps ad onto the court plane into the frame.

```python
class AdPlacer:
    def warp(self, ad_rgba, H, placement_spec) -> dict:
        return {"warped_rgba": ..., "warped_mask": ...}
```

### D) `Compositor`

Composites with occlusions + blending.

```python
class Compositor:
    def composite(self, frame, warped_rgba, warped_mask, occ_mask, blend_cfg) -> dict:
        return {"frame_out": ..., "debug": dict}
```

### E) `TemporalStabilizer`

Smooths `H` (and optionally masks).

```python
class TemporalStabilizer:
    def update_H(self, H, conf) -> np.ndarray:
        ...
```

---

## Milestones (what you present to the team)

### MVP demo (1–2 weeks max)

* Court homography works on a sample broadcast clip.
* Ad is stable enough (smoothing + cut reset).
* Players occlude ad (even roughly).
* Output video saved with debug overlays toggled.

### Next demo (upgrade realism)

* Better occlusion edges (less halo).
* Shadow-preserving blend (ad darkens under shadows).
* Optional ball occlusion.

---

## Atomic step plan (Cursor-friendly)

### Step 0 — New repo scaffold (1 task)

Create a new repo (e.g., `tennis-virtual-ads/`) with:

```
src/
  pipeline/
    calibrators/
    maskers/
    placer/
    compositor/
    temporal/
  io/
  utils/
configs/
scripts/
tests/
docs/
```

Add:

* `pyproject.toml` or `requirements.txt`
* `configs/default.yaml`
* `scripts/run_video.py` (entry point)
* `docs/PLAN.md` (this doc)

**Definition of done:** running `python scripts/run_video.py ...` loads a video and iterates frames (no ML yet).

---

### Step 1 — Video I/O + deterministic frame loop (1 task)

Implement:

* `VideoReader` (cv2)
* `VideoWriter` (cv2)
* frame iterator with:

  * `--max_frames`
  * `--start_frame`
  * `--stride` (for fast dev)
  * `--resize` option

**Done:** Can copy input→output video, and optionally overlay frame index.

---

### Step 2C — Video integration (completed)

**Goal:** Run TennisCourtDetectorCalibrator on every frame of a video and
produce an output with court overlay.

**What was implemented:**

* `run_video.py` extended with `--calibrator tennis_court_detector`,
  `--weights_path`, `--calib_conf_threshold`, `--draw_mode {overlay,keypoints,none}`.
* Per-frame status HUD: "CALIB OK conf=... kp=N/14 err=...px" (green) or
  "NO CALIB conf=... kp=N/14" (red).
* Shared drawing module `src/tennis_virtual_ads/utils/draw.py` -- extracted
  from debug_calibrator_image.py so both scripts reuse the same helpers.
* Lazy calibrator import: torch is only loaded when `--calibrator tennis_court_detector`
  is selected.
* End-of-run stats: accepted/rejected frame counts and accept rate.

**Run command:**

```bash
uv run python scripts/run_video.py \
    djokovic-10-sec.mp4 output_overlay.mp4 \
    --calibrator tennis_court_detector \
    --draw_mode overlay
```

**Known limitations:**

* No temporal smoothing -- overlay may "swim" frame-to-frame.
* No scene-cut detection -- if the camera changes, stale H may flash briefly.
* CPU-only inference is slow (~1-3 fps); GPU speeds this up significantly.

---

### Step 3A — Temporal keypoint smoothing (completed)

**Goal:** Reduce frame-to-frame overlay jitter by smoothing detected
keypoints over time and recomputing the homography from the smoothed
keypoints.

**What was implemented:**

* `KeypointSmoother` class in `src/tennis_virtual_ads/pipeline/temporal/keypoint_smoother.py`
  using per-keypoint exponential moving average (EMA).
* Configurable `alpha` (default 0.7): higher = more responsive, lower = smoother.
* Error spike detection: if reprojection error exceeds `factor * median(recent errors)`,
  the smoother auto-resets to prevent drift after bad detections or scene changes.
* Integration in `run_video.py` with new CLI flags:
  `--smooth_keypoints`, `--kp_smooth_alpha`, `--reset_on_err_spike`,
  `--no_reset_on_err_spike`, `--err_spike_factor`.
* Third HUD line shows smoothing status: "SMOOTH=ON err=X.Xpx" (cyan)
  or "SMOOTH=RESET (spike detected)" (yellow).
* End-of-run stats include reset count.
* `JitterTracker` in `src/tennis_virtual_ads/pipeline/temporal/jitter_tracker.py`
  quantifies overlay stability using projected-point acceleration (MPPA).
  Reports mean, p95, and max acceleration in pixels at end-of-run.
  When smoothing is enabled, tracks both raw and smoothed H for direct comparison.
  Disable with `--no_jitter_tracker`.

**Run commands (comparison):**

```bash
# Baseline (no smoothing):
uv run python scripts/run_video.py \
    djokovic-10-sec.mp4 output_baseline.mp4 \
    --calibrator tennis_court_detector \
    --draw_mode overlay

# With smoothing (mild):
uv run python scripts/run_video.py \
    djokovic-10-sec.mp4 output_smooth_07.mp4 \
    --calibrator tennis_court_detector \
    --draw_mode overlay \
    --smooth_keypoints --kp_smooth_alpha 0.7

# With smoothing (aggressive):
uv run python scripts/run_video.py \
    djokovic-10-sec.mp4 output_smooth_03.mp4 \
    --calibrator tennis_court_detector \
    --draw_mode overlay \
    --smooth_keypoints --kp_smooth_alpha 0.3
```

Compare the "Jitter (raw H)" and "Jitter (smoothed H)" lines in the
end-of-run output.  Lower `mean_accel` = more stable overlay.

**Architecture notes:**

* Smoothing is a post-processing layer that sits between the calibrator
  and the drawing step. The calibrator still runs independently per frame.
* `KeypointSmoother` is stateless with respect to the calibrator -- it
  only receives the `(14, 2)` keypoints array and optionally the
  reprojection error.
* After smoothing, the homography is recomputed from smoothed keypoints
  using the same `get_trans_matrix` best-of-12 selector.
* If smoothing is disabled (`--smooth_keypoints` not passed), the
  pipeline behaves exactly as before -- zero performance/behavior change.

**Known limitations:**

* No scene-cut detection -- the smoother relies on error spike detection
  as a proxy, which may be slow to react on hard cuts (Step 3B).
* EMA has no "forgetting" of keypoints that disappear -- if a keypoint
  stops being detected, the smoothed value carries forward indefinitely.

---

### Step 4A — Ad warp + naive composite (completed)

**Goal:** Warp an RGBA ad image onto the court surface using the
homography and alpha-composite it into the output video.

**What was implemented:**

* `PlacementSpec` TypedDict + `compute_ad_court_corners()` in
  `src/tennis_virtual_ads/pipeline/placer/placement.py` -- converts
  anchor name + width/height/offset ratios into 4 court-reference
  corners.
* `AdPlacer` class in `src/tennis_virtual_ads/pipeline/placer/ad_placer.py`
  with `warp()` (perspective transform) and `composite()` (alpha blend).
* Integration in `run_video.py` with CLI flags: `--ad_enable`,
  `--ad_image_path`, `--ad_anchor`, `--ad_width_ratio`,
  `--ad_height_ratio`, `--ad_y_offset_ratio`.
* Config defaults in `configs/default.yaml` under `ad:` section.
* HUD line: "AD=ON anchor=near_baseline_center (smooth H)" in magenta.
* Uses smoothed H when `--smooth_keypoints` is enabled; raw H otherwise.
* Ad is skipped on frames where calibration fails (no H available).

**Run command:**

```bash
uv run python scripts/run_video.py \
    djokovic-10-sec.mp4 output_with_ad.mp4 \
    --calibrator tennis_court_detector \
    --draw_mode overlay --smooth_keypoints \
    --ad_enable --ad_image_path assets/test_ad.png
```

**Architecture notes:**

* Placement is defined in court-reference coordinates (the same coordinate
  system used by `CourtReference`).  The 4 corners are computed once at
  startup, then projected through H each frame via `cv2.perspectiveTransform`.
* `AdPlacer` is stateless and reusable across frames.
* Alpha compositing is naive: `frame = frame*(1-alpha) + ad*alpha`.
  No occlusion masking yet -- the ad draws *over* players.
* Anchor registry is extensible: add new entries to `_ANCHOR_REGISTRY`
  in `placement.py` to support other court positions.

**Known limitations:**

* No occlusion handling -- ad is drawn over players, ball, rackets.
* Only one ad at a time (single placement spec).
* Ad appears flat -- no shadow/lighting adaptation.

---

### Step 5A — OcclusionMasker v1: players only (completed)

**Goal:** Generate a per-frame occlusion mask for people (players, ball
kids, umpire) and use it to prevent the ad from drawing over them.

**What was implemented:**

* `OcclusionMasker` ABC + `OcclusionMaskerResult` TypedDict in
  `src/tennis_virtual_ads/pipeline/maskers/base.py` — mirrors the
  `CourtCalibrator` / `CalibrationResult` pattern.
* `PersonMasker` in `src/tennis_virtual_ads/pipeline/maskers/person_masker.py`
  using torchvision `maskrcnn_resnet50_fpn` (COCO-pretrained, person
  class = label 1).  Per-instance masks are merged via max (union) and
  binarised at 0.5.
* Masker integration in `run_video.py` with CLI flags:
  `--masker {none, person}`, `--masker_conf_threshold`,
  `--mask_dilate_px`, `--mask_debug`.
* Occlusion compositing: `effective_alpha = warped_mask * (1 - occ_mask)`
  so players remain visible through the ad.
* Dilation via `cv2.dilate` with an elliptical kernel to extend the
  mask slightly and cover rackets near the body.
* HUD line shows "MASK=person persons=N" in orange.
* `--mask_debug` renders a red-tinted mask preview in the bottom-right
  corner of the output video.
* `[occlusion]` optional extras group added to `pyproject.toml`.

**Dependencies:**

Requires `torch` + `torchvision` (same as `[calibration]`).  Install:

```bash
uv pip install -e ".[occlusion]"
```

**Note:** On first run, torchvision auto-downloads Mask R-CNN pretrained
weights (~170 MB).  A log message is emitted during loading.

**Run command:**

```bash
uv run python scripts/run_video.py \
    djokovic-10-sec.mp4 output_with_occlusion.mp4 \
    --calibrator tennis_court_detector \
    --draw_mode overlay --smooth_keypoints \
    --ad_enable --ad_image_path assets/test_ad.png \
    --masker person --mask_debug
```

**Architecture notes:**

* The masker runs independently per frame before ad compositing.  It
  does not know about homography or placement — it only produces a
  pixel mask.
* The `effective_alpha` multiplication is the only coupling point
  between the masker and the ad placer, keeping both modules fully
  decoupled.
* Masking is purely optional — when `--masker none` (default), the
  pipeline behaves exactly as before with zero overhead.

**Known limitations:**

* Mask R-CNN is ~200–500 ms/frame on CPU.  GPU recommended for
  interactive speeds.
* No ball, net, or shadow handling yet (Step 5B).
* Mask edges may show slight halo artifacts on fast-moving limbs.
* Only one masker type (`person`) available; the factory is extensible.

---

### Step 2B — Single-image calibrator proof (completed)

**Goal:** Prove the TennisCourtDetector model works end-to-end on a single image.

**What was implemented:**

* `TennisCourtDetectorCalibrator` — wraps BallTrackerNet (14-keypoint heatmap
  model) behind the `CourtCalibrator` interface.
* `scripts/debug_calibrator_image.py` — CLI to run calibration on one image
  and save annotated output.
* Vendored court reference + homography logic in `_tcd_adapted/` subpackage
  (stripped unused matplotlib/scipy/sympy dependencies).
* BallTrackerNet model imported via `importlib` from sibling
  `TennisCourtDetector/` repo (no `sys.path` pollution).

**Weights:**

* Download from: <https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG>
* Place at: `weights/tennis_court_detector.pt`
* Weights are git-ignored (large binary).

**Run command:**

```bash
uv run python scripts/debug_calibrator_image.py \
    --image_path ../tennis_court_detection/test_images/tennis_pic_01.png \
    --output_path output_debug.png
```

**Dependencies (one-time install):**

```bash
uv pip install torch
```

**Known limitations:**

* Only tested on behind-baseline broadcast angles (model training data).
* No `refine_kps` postprocessing (would need sympy; deferred to later task).
* No video processing yet — single image only.
* No temporal stabilisation.

---

### Step 2A — Calibrator interface + dummy wiring (1 task)

Define:

* `CourtCalibrator` ABC with `estimate(frame) -> CalibrationResult`
* `CalibrationResult` TypedDict (`H`, `conf`, `keypoints`, `debug`)
* `DummyCalibrator` (returns `H=None`, `conf=0.0` always)
* `--calibrator` CLI flag in `run_video.py`

**Done:** `run_video.py --calibrator dummy` produces output with "CALIB: dummy conf=0.00" overlay.

---

### Step 2B — Ad representation + placement spec (1 task)

Define:

* `ad_rgba` loading (PNG with alpha)
* `placement_spec` (how you choose location)

  * MVP: define ad in **court reference coordinates** (e.g., rectangle using known court model coords)
  * Example spec:

    * `"anchor": "near_baseline_center"`
    * `"width_m": 6.0`
    * `"height_m": 2.0`
    * `"offset_m": (0.0, 1.0)` etc.

**Done:** Ad can be warped once you have `H`.

---

### Step 3 — CourtCalibrator v1 (TennisCourtDetector) (2–3 tasks)

**Goal:** from a single frame, get `H` reliably.

Tasks:

1. Create a wrapper `TennisCourtDetectorCalibrator`:

   * load pretrained weights
   * run inference on frame
   * get keypoints + compute `H` using their homography module
2. Standardize orientation:

   * decide: `H_court_to_img`
   * validate by projecting known court points and drawing lines
3. Add confidence score:

   * based on number of detected points
   * reprojection error from best-config selection

**Done:** `scripts/run_video.py` can draw court overlay on top of video and write output.

---

### Step 4 — Temporal stabilizer for homography (1–2 tasks)

Implement:

* exponential moving average on `H` parameters OR smooth keypoints then recompute `H`
* cut detection:

  * simplest: if `conf` drops hard or reprojection error spikes, reset
  * optionally integrate CourtCheck scene detection later

**Done:** overlay no longer "swims" frame-to-frame on steady shots.

---

### Step 5 — Ad warp + naive composite (1 task)

Implement `AdPlacer.warp()` using OpenCV:

* compute 4 target points in court coords → map to image via `H`
* `cv2.warpPerspective` the ad into the frame
* produce `warped_mask` from alpha channel

Implement `Compositor` naive:

* `frame_out = frame*(1-mask) + warped_rgb*mask`

**Done:** ad looks placed and stable (but goes over players).

---

### Step 6 — OcclusionMasker v1 (players only) (2–3 tasks)

Pick a simple baseline you can run today:

* Option A: torchvision Mask R-CNN pre-trained (person class)
* Option B: ultralytics YOLO-seg (person)
* Option C: a matting model (MODNet) later for cleaner edges

MVP:

* return `occ_mask` for person pixels only
* optionally expand mask slightly (dilation) to cover racket bits

**Done:** players occlude the ad.

---

### Step 7 — Composite with occlusion (1 task)

Modify compositor:

* effective_mask = warped_mask * (1 - occ_mask)

**Done:** the ad disappears under players.

---

### Step 8 — Debug views (1 task)

Add `--debug` flags to write:

* side-by-side: original | court overlay | occ mask | final
* or write separate videos

**Done:** team can see *which component is failing* when something looks wrong.

---

## Upgrade track (post-MVP)

### U1 — Better occlusion classes

* Add ball mask:

  * easiest: track ball point and carve small circle mask
  * later: TrackNet-based
* Add "other foreground":

  * ball kids
  * umpires
  * net region
* Improve racket coverage:

  * train or heuristic dilation near hands

### U2 — Shadow-preserving blend (high realism impact)

Goal: shadows remain visible over the ad.
Approach:

* estimate "shading field" on court from original frame (low-frequency illumination)
* modulate ad brightness by that field before compositing
* minimal viable:

  * blur original frame heavily to estimate illumination
  * compute ratio between illumination and court average
  * scale ad V channel by ratio

### U3 — Better cut handling

* integrate PySceneDetect as optional module (CourtCheck already uses it)
* on cut: reset stabilizer, reinitialize calibrator

### U4 — Performance (towards 30 fps)

* batch inference or async pipeline
* export models to ONNX/TensorRT
* minimize CPU copies
* run segmentation every N frames and propagate with optical flow (optional)

---

## Decision points (so you don't get stuck)

### Court calibration choice

Start with **TennisCourtDetector** because:

* dataset + weights exist
* homography correction built-in
  Then optionally compare:
* `CourtCheck` homography selection (might be cleaner integration)
* `tennis_court_detection` for sanity checks

### Occlusion baseline

Don't wait for tennis-specific training.
Start with generic person segmentation and make the pipeline work end-to-end.

---

## What you should NOT do yet (avoid rabbit holes)

* Don't start with ball/racket/net perfection.
* Don't optimize to 30 fps before you have correct geometry + occlusion.
* Don't refactor repos into your repo; wrap them behind interfaces.
---
