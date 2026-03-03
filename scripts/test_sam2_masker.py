#!/usr/bin/env python3
"""Quick smoke test for SAM2Masker on a single frame.

Usage
-----
    # With a test video:
    uv run python scripts/test_sam2_masker.py assets/videos/djokovic-10-sec.mp4

    # With a specific frame index:
    uv run python scripts/test_sam2_masker.py assets/videos/djokovic-10-sec.mp4 --frame 30

    # Save mask visualisation:
    uv run python scripts/test_sam2_masker.py assets/videos/djokovic-10-sec.mp4 --save mask_out.png

Prints: timing, mask shape, instance count, GPU memory usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Make src/ importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_sam2")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2Masker smoke test")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--frame", type=int, default=30, help="Frame index to test (default: 30)")
    parser.add_argument("--save", type=str, default=None, help="Save mask as PNG to this path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM2 checkpoint (default: weights/sam2.1_hiera_small.pt)",
    )
    args = parser.parse_args()

    # Read a single frame.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", args.video)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Failed to read frame %d from %s", args.frame, args.video)
        sys.exit(1)

    logger.info("Frame %d: shape=%s dtype=%s", args.frame, frame.shape, frame.dtype)

    # Import and create masker.
    from tennis_virtual_ads.pipeline.maskers.sam2_masker import SAM2Masker

    masker_kwargs: dict[str, Any] = {}
    if args.checkpoint:
        masker_kwargs["checkpoint_path"] = args.checkpoint

    masker = SAM2Masker(**masker_kwargs)

    # Run masking.
    t0 = time.perf_counter()
    result = masker.mask(frame)
    elapsed = time.perf_counter() - t0

    mask = result["mask"]
    conf = result["conf"]
    debug = result["debug"]

    logger.info("SAM2 mask completed in %.3f s", elapsed)
    logger.info(
        "  mask shape=%s  dtype=%s  min=%.2f  max=%.2f  nonzero=%d",
        mask.shape,
        mask.dtype,
        float(np.min(mask)),
        float(np.max(mask)),
        int(np.count_nonzero(mask)),
    )
    logger.info("  confidence=%.3f", conf)
    logger.info("  instance_count=%s", debug.get("instance_count", "N/A"))
    logger.info("  prompt_method=%s", debug.get("prompt_method", "N/A"))

    # GPU memory.
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info("  GPU memory: allocated=%.0f MB  reserved=%.0f MB", allocated, reserved)
    except ImportError:
        pass

    # Save mask visualisation.
    if args.save:
        # Create a red-tinted overlay for visualisation.
        vis = frame.copy()
        mask_bool = mask > 0.5
        vis[mask_bool, 2] = np.clip(vis[mask_bool, 2].astype(np.int16) + 100, 0, 255).astype(
            np.uint8
        )
        cv2.imwrite(args.save, vis)
        logger.info("Saved mask visualisation to %s", args.save)

    logger.info("Done!")


if __name__ == "__main__":
    main()
