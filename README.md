# Tennis Virtual Ads

Virtual advertisement overlay for tennis broadcasts.
Takes a broadcast video, estimates court geometry, and composites an ad onto the court surface — with player occlusion so the result looks natural.

> **Status:** Step 0/1 complete — scaffold + video I/O entrypoint.

---

## EC2 Setup (g4dn.xlarge, Ubuntu 24.04)

Setup requires a reboot (NVIDIA driver), so it's split into two scripts.

### Quick Setup
```
# Connect
ssh -A -i ~/repositories/CS265-mlsys-project/cs265-ed25519.pem ubuntu@ec2-3-236-216-236.compute-1.amazonaws.com

# Phase 1 (first time only — ~2-3 min, then reboots)
mkdir -p ~/capstone-repos && cd ~/capstone-repos
git clone git@github.com:enriquedlh97/tennis-virtual-ads.git
cd tennis-virtual-ads
bash scripts/setup_ec2.sh

# Phase 2 (after reconnecting — ~30s)
cd ~/capstone-repos/tennis-virtual-ads
bash scripts/setup_ec2_post_reboot.sh
```

### 1. Connect and clone

```bash
ssh -A -i ~/repositories/CS265-mlsys-project/cs265-ed25519.pem ubuntu@ec2-3-236-216-236.compute-1.amazonaws.com
```

```bash
mkdir -p ~/capstone-repos && cd ~/capstone-repos
git clone git@github.com:enriquedlh97/tennis-virtual-ads.git
cd tennis-virtual-ads
```

### 2. Run Phase 1 (installs system deps, NVIDIA driver, uv, clones sibling repos → reboots)

```bash
bash scripts/setup_ec2.sh
```

This installs:
- Git config (author name/email) + vim + terminal prompt with git branch
- System packages (`libgl1`, `libsm6`, `libxrender1` for OpenCV)
- NVIDIA driver 570 (for T4 GPU)
- `uv` (Python package manager)
- Sibling repos (`CourtCheck`, `tennis_court_detection`, `tennis-court-tracker`, `TennisCourtDetector`)

The instance **reboots automatically** at the end.

### 3. Reconnect and run Phase 2 (verifies GPU, installs Python deps, runs checks)

```bash
ssh -A -i ~/repositories/CS265-mlsys-project/cs265-ed25519.pem ubuntu@ec2-3-236-216-236.compute-1.amazonaws.com
cd ~/capstone-repos/tennis-virtual-ads
bash scripts/setup_ec2_post_reboot.sh
```

This verifies:
- `nvidia-smi` sees the T4 GPU
- `uv sync --all-extras` installs all Python dependencies
- `pytest`, `ruff`, `mypy` all pass

### Workspace layout after setup

```
~/capstone-repos/
├── tennis-virtual-ads/          ← our repo
├── tennis-court-tracker/        ← sibling (court tracking)
├── tennis_court_detection/      ← sibling (classical CV)
├── CourtCheck/                  ← sibling (homography + tracking)
└── TennisCourtDetector/         ← sibling (heatmap keypoints)
```

---

## Quickstart (local or after EC2 setup)

### 1. Install dependencies

```bash
uv sync --all-extras
```

Creates a virtual environment, installs runtime deps (`opencv-python`, `PyYAML`) and dev deps (`pytest`, `ruff`, `mypy`, `coverage`, `pre-commit`).

### 2. Run on a local video

```bash
uv run python scripts/run_video.py path/to/input.mp4 output.mp4
```

Reads `input.mp4`, overlays the frame index on every frame, and writes `output.mp4`.

### 3. Useful flags

```bash
# First 200 frames, starting at frame 500, every 2nd frame
uv run python scripts/run_video.py input.mp4 output.mp4 \
    --start_frame 500 \
    --max_frames 200 \
    --stride 2

# Downscale to 640×360 for faster iteration
uv run python scripts/run_video.py input.mp4 output.mp4 --resize 640x360
```

### 4. Dev commands

```bash
uv run pytest tests/ -v              # Run tests
uv run bash scripts/lint.sh          # Lint + type check (no auto-fix)
uv run bash scripts/format.sh        # Auto-fix + format
uv run bash scripts/test.sh          # Tests with coverage report
uv run pre-commit run --all-files    # Run all pre-commit hooks
```

---

## Project structure

```
tennis-virtual-ads/
├── configs/
│   └── default.yaml                # Default CLI flag values
├── scripts/
│   ├── run_video.py                # CLI entrypoint
│   ├── setup_ec2.sh                # EC2 setup — phase 1 (pre-reboot)
│   ├── setup_ec2_post_reboot.sh    # EC2 setup — phase 2 (post-reboot)
│   ├── lint.sh                     # mypy + ruff check
│   ├── format.sh                   # ruff fix + format
│   └── test.sh                     # pytest + coverage
├── src/tennis_virtual_ads/
│   ├── io/
│   │   └── video.py                # VideoReader / VideoWriter
│   ├── pipeline/                   # (future) calibrators, maskers, placer, compositor
│   └── utils/                      # (future) geometry, drawing, config helpers
├── tests/
│   └── test_smoke.py               # Smoke tests
├── .pre-commit-config.yaml         # Pre-commit hooks
├── pyproject.toml                  # Project metadata & dependencies (uv)
├── PLAN.md                         # Full project plan & architecture
└── README.md
```

---

## Architecture (coming next)

See [PLAN.md](PLAN.md) for the full pipeline design:

- **CourtCalibrator** — homography estimation per frame
- **OcclusionMasker** — player / foreground segmentation
- **AdPlacer** — warp ad onto court plane
- **Compositor** — blend with occlusion awareness
- **TemporalStabilizer** — smooth homography over time
