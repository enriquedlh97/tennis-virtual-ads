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
#   4. Installs pre-commit git hooks
#   5. Runs smoke tests, linter, and type checker
#   6. Prints success summary
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
info "1/5  Verifying NVIDIA GPU"

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
info "2/5  Verifying uv"

if command -v uv &> /dev/null; then
    pass "uv: $(uv --version)"
else
    fail "uv not found on PATH. Re-run setup_ec2.sh or install manually."
fi

# --------------------------------------------------------------------------
# 3. Install Python dependencies + pre-commit hooks
# --------------------------------------------------------------------------
info "3/5  Installing Python dependencies"

uv sync --all-extras
pass "uv sync complete (venv created at .venv/)"

info "4/5  Installing pre-commit hooks"

uv run pre-commit install
pass "Pre-commit hooks installed (will run on every git commit)"

# --------------------------------------------------------------------------
# 5. Run checks (tests + lint + type check)
# --------------------------------------------------------------------------
info "5/5  Running checks"

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
info "🎉  Setup complete!"

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
