# PLAN.md — Tennis Virtual Ad Overlay Baseline (Single Broadcast Feed)

## Progress

- [x] **Step 0** — Repo scaffold (`pyproject.toml`, `configs/`, `scripts/run_video.py`, project layout)
- [x] **Step 1** — Video I/O + deterministic frame loop (`VideoReader`, `VideoWriter`, frame-index overlay)
- [x] **Step 2A** — Calibrator interface + dummy wiring (`CourtCalibrator` ABC, `DummyCalibrator`, `--calibrator` flag)
- [ ] **Step 2B** — Ad representation + placement spec
- [ ] **Step 3** — CourtCalibrator v1 (TennisCourtDetector wrapper)
- [ ] **Step 4** — Temporal stabilizer for homography
- [ ] **Step 5** — Ad warp + naive composite
- [ ] **Step 6** — OcclusionMasker v1 (players only)
- [ ] **Step 7** — Composite with occlusion
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
