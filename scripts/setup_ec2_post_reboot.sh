#!/usr/bin/env bash
# ==========================================================================
# Tennis Virtual Ads — EC2 g4dn.xlarge Setup (Phase 2: Post-Reboot)
#
# Usage:
#   ssh -A -i <key>.pem ubuntu@<ec2-host>
#   cd ~/capstone-repos/tennis-virtual-ads
#   bash scripts/setup_ec2_post_reboot.sh
#
# What this does:
#   1. Verifies NVIDIA GPU is visible
#   2. Verifies uv is on PATH
#   3. Installs Python dependencies (uv sync → creates .venv/)
#   4. Downloads model weights (TennisCourtDetector from Google Drive)
#   5. Downloads test videos into assets/ (from Google Drive zip)
#   6. Installs pre-commit git hooks
#   7. Runs smoke tests, linter, and type checker
#   8. Prints success summary
# ==========================================================================

set -euo pipefail

# Ensure uv is on PATH (in case .bashrc hasn't been sourced yet)
export PATH="$HOME/.local/bin:$PATH"

# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
info() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
    echo ""
}

pass() { echo "  ✅  $1"; }
fail() { echo "  ❌  $1"; exit 1; }

# --------------------------------------------------------------------------
# 1. Verify GPU
# --------------------------------------------------------------------------
info "1/8  Verifying NVIDIA GPU"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    pass "GPU: $GPU_NAME | Driver: $DRIVER_VERSION"
else
    fail "nvidia-smi not found. Did setup_ec2.sh complete and the instance reboot?"
fi

# --------------------------------------------------------------------------
# 2. Verify uv
# --------------------------------------------------------------------------
info "2/8  Verifying uv"

if command -v uv &> /dev/null; then
    pass "uv: $(uv --version)"
else
    fail "uv not found on PATH. Re-run setup_ec2.sh or install manually."
fi

# --------------------------------------------------------------------------
# 3. Install Python dependencies + pre-commit hooks
# --------------------------------------------------------------------------
info "3/8  Installing Python dependencies"

uv sync --all-extras
pass "uv sync complete (venv created at .venv/)"

# --------------------------------------------------------------------------
# 4. Download model weights
# --------------------------------------------------------------------------
info "4/8  Downloading model weights"

bash scripts/download_weights.sh
pass "Model weights ready"

# --------------------------------------------------------------------------
# 5. Download test videos
# --------------------------------------------------------------------------
info "5/8  Downloading test videos"

VIDEOS_ZIP="assets/videos.zip"
VIDEOS_DIR="assets/videos"
VIDEOS_GDRIVE_ID="1tawxvTSqW4It6nDYQ7c0_6zSjBBSzZY7"

if [ -d "$VIDEOS_DIR" ] && [ "$(ls -A "$VIDEOS_DIR" 2>/dev/null)" ]; then
    echo "  Test videos already exist in $VIDEOS_DIR — skipping download"
    pass "Test videos ready ($(ls "$VIDEOS_DIR" | wc -l) files)"
else
    echo "  Downloading test videos from Google Drive..."
    echo "  File ID: $VIDEOS_GDRIVE_ID"
    uv run --with gdown gdown "$VIDEOS_GDRIVE_ID" -O "$VIDEOS_ZIP"

    if [ ! -f "$VIDEOS_ZIP" ]; then
        fail "Download failed — $VIDEOS_ZIP not found"
    fi

    if ! command -v unzip &> /dev/null; then
        echo "  Installing unzip..."
        sudo apt install -y unzip
    fi

    echo "  Unzipping into $VIDEOS_DIR ..."
    mkdir -p "$VIDEOS_DIR"
    unzip -o "$VIDEOS_ZIP" -d "$VIDEOS_DIR"
    rm -f "$VIDEOS_ZIP"

    pass "Test videos downloaded ($(ls "$VIDEOS_DIR" | wc -l) files in $VIDEOS_DIR)"
fi

info "6/8  Installing pre-commit hooks"

uv run pre-commit install
pass "Pre-commit hooks installed (will run on every git commit)"

# --------------------------------------------------------------------------
# 5. Run checks (tests + lint + type check)
# --------------------------------------------------------------------------
info "7/8  Running checks"

echo "  --- pytest ---"
uv run pytest tests/ -v
pass "Tests passed"

echo ""
echo "  --- ruff ---"
uv run ruff check src scripts
pass "Ruff lint passed"

echo ""
echo "  --- mypy ---"
uv run mypy src scripts
pass "Mypy type check passed"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
info "8/8  Setup complete!"

PYTHON_VERSION=$(uv run python --version 2>&1)

echo "  Instance:  $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Driver:    $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"
echo "  Python:    $PYTHON_VERSION"
echo "  uv:        $(uv --version)"
echo ""
echo "  Workspace: ~/capstone-repos/"
ls -1 "$HOME/capstone-repos/"
echo ""
echo "  Ready to go! Try:"
echo ""
echo "    cd ~/capstone-repos/tennis-virtual-ads"
echo "    uv run python scripts/run_video.py input.mp4 output.mp4"
echo ""
