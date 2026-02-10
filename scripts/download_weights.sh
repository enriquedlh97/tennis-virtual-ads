#!/usr/bin/env bash
# ==========================================================================
# Tennis Virtual Ads — Download Model Weights
#
# Usage:
#   bash scripts/download_weights.sh
#
# What this does:
#   1. Downloads TennisCourtDetector weights from Google Drive
#      → weights/tennis_court_detector.pt (~178 MB)
#   2. Skips download if file already exists (idempotent)
#   3. Verifies file size after download
#
# Note: Mask R-CNN weights (used by PersonMasker) are auto-downloaded
# by torchvision on first run (~170 MB, cached in ~/.cache/torch/).
# No manual download needed for those.
#
# Dependencies: uv (uses `uv run --with gdown` for ephemeral install)
# ==========================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# TennisCourtDetector weights
# Source: https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG
GDRIVE_FILE_ID="1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
OUTPUT_PATH="weights/tennis_court_detector.pt"
MIN_SIZE_BYTES=100000000  # ~100 MB minimum (actual file is ~178 MB)

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

# --------------------------------------------------------------------------
# Ensure weights/ directory exists
# --------------------------------------------------------------------------
mkdir -p weights

# --------------------------------------------------------------------------
# Download TennisCourtDetector weights
# --------------------------------------------------------------------------
info "Downloading model weights"

if [ -f "$OUTPUT_PATH" ]; then
    FILE_SIZE=$(stat --printf="%s" "$OUTPUT_PATH" 2>/dev/null || stat -f%z "$OUTPUT_PATH" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -gt "$MIN_SIZE_BYTES" ]; then
        echo "  TennisCourtDetector weights already exist: $OUTPUT_PATH ($(( FILE_SIZE / 1024 / 1024 )) MB)"
        echo "  Skipping download."
    else
        echo "  File exists but looks too small (${FILE_SIZE} bytes) — re-downloading..."
        rm -f "$OUTPUT_PATH"
        echo "  Downloading from Google Drive (file ID: $GDRIVE_FILE_ID) ..."
        uv run --with gdown gdown "$GDRIVE_FILE_ID" -O "$OUTPUT_PATH"
    fi
else
    echo "  Downloading TennisCourtDetector weights from Google Drive..."
    echo "  File ID: $GDRIVE_FILE_ID"
    echo "  Output:  $OUTPUT_PATH"
    echo ""
    uv run --with gdown gdown "$GDRIVE_FILE_ID" -O "$OUTPUT_PATH"
fi

# --------------------------------------------------------------------------
# Verify download
# --------------------------------------------------------------------------
if [ ! -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "  ERROR: Download failed — $OUTPUT_PATH not found."
    echo "  Try downloading manually from:"
    echo "    https://drive.google.com/file/d/$GDRIVE_FILE_ID"
    echo "  and place the file at: $OUTPUT_PATH"
    exit 1
fi

FILE_SIZE=$(stat --printf="%s" "$OUTPUT_PATH" 2>/dev/null || stat -f%z "$OUTPUT_PATH" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt "$MIN_SIZE_BYTES" ]; then
    echo ""
    echo "  ERROR: Downloaded file looks too small (${FILE_SIZE} bytes)."
    echo "  Expected at least $(( MIN_SIZE_BYTES / 1024 / 1024 )) MB."
    echo "  The Google Drive link may have changed. Try downloading manually from:"
    echo "    https://drive.google.com/file/d/$GDRIVE_FILE_ID"
    rm -f "$OUTPUT_PATH"
    exit 1
fi

echo ""
echo "  TennisCourtDetector weights: $OUTPUT_PATH ($(( FILE_SIZE / 1024 / 1024 )) MB)"
echo ""
echo "  Note: Mask R-CNN weights (for --masker person) will be"
echo "  auto-downloaded by torchvision on first run (~170 MB)."
echo ""
echo "  Weights ready!"
