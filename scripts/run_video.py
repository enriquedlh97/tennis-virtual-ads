#!/usr/bin/env python3
"""CLI entrypoint — read a video, overlay frame indices, write output.

Usage
-----
    python scripts/run_video.py input.mp4 output.mp4
    python scripts/run_video.py input.mp4 output.mp4 --max_frames 300 --resize 640x360

Config defaults are loaded from ``configs/default.yaml``; any CLI flag
overrides the corresponding config value.
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
import yaml

# ---------------------------------------------------------------------------
# Make the ``src/`` tree importable when running the script directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tennis_virtual_ads.io.video import VideoReader, VideoWriter  # noqa: E402

logger = logging.getLogger("run_video")

DEFAULTS_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict."""
    if config_path.exists():
        with open(config_path) as config_file:
            return yaml.safe_load(config_file) or {}
    logger.warning("Config file not found: %s — using built-in defaults.", config_path)
    return {}


def parse_resize(value: str | None) -> tuple[int, int] | None:
    """Parse a ``'WIDTHxHEIGHT'`` string into an ``(int, int)`` tuple.

    Returns ``None`` when *value* is ``None``.
    """
    if value is None:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid resize format '{value}'. Expected 'WIDTHxHEIGHT', e.g. '1280x720'."
        )
    return int(parts[0]), int(parts[1])


def overlay_frame_index(frame: np.ndarray, frame_index: int) -> None:
    """Draw the frame index onto the top-left of *frame* (mutates in-place)."""
    text = f"Frame {frame_index}"
    _frame_height, frame_width = frame.shape[:2]

    # Scale font relative to frame width so text is readable at any resolution
    font_scale = max(0.5, frame_width / 1280.0)
    thickness = max(1, int(font_scale * 2))
    position = (10, 30 + int(font_scale * 10))

    # Black outline for contrast, then white fill
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Tennis Virtual Ads — Video I/O scaffold.  "
            "Reads a video, overlays frame indices, writes output."
        ),
    )
    parser.add_argument("input", help="Path to input video file (e.g. match.mp4)")
    parser.add_argument("output", help="Path to output video file (e.g. output.mp4)")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=None,
        help="Frame index to start from, 0-based (default: 0)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Process every N-th frame (default: 1)",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize frames to WIDTHxHEIGHT, e.g. '1280x720' (default: original)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_argument_parser()
    args = parser.parse_args()

    # --- Merge config defaults with CLI overrides -------------------------
    config_path = Path(args.config) if args.config else DEFAULTS_CONFIG_PATH
    config = load_config(config_path)
    logger.info("Loaded config from %s", config_path)

    start_frame: int = (
        args.start_frame if args.start_frame is not None else config.get("start_frame", 0)
    )
    max_frames: int | None = (
        args.max_frames if args.max_frames is not None else config.get("max_frames", None)
    )
    stride: int = args.stride if args.stride is not None else config.get("stride", 1)
    resize_string: str | None = (
        args.resize if args.resize is not None else config.get("resize", None)
    )
    resize: tuple[int, int] | None = parse_resize(resize_string)

    logger.info(
        "Settings — start_frame=%d  max_frames=%s  stride=%d  resize=%s",
        start_frame,
        max_frames,
        stride,
        resize,
    )

    # --- Process video ----------------------------------------------------
    wall_clock_start = time.perf_counter()

    with VideoReader(
        args.input,
        start_frame=start_frame,
        max_frames=max_frames,
        stride=stride,
        resize=resize,
    ) as reader:
        output_width = resize[0] if resize else reader.width
        output_height = resize[1] if resize else reader.height

        with VideoWriter(
            args.output,
            fps=reader.fps,
            width=output_width,
            height=output_height,
        ) as writer:
            for frame_index, frame in reader:
                overlay_frame_index(frame, frame_index)
                writer.write(frame)

                if writer.frames_written % 100 == 0:
                    elapsed = time.perf_counter() - wall_clock_start
                    logger.info(
                        "Progress: %d frames written  (%.1f s elapsed, ~%.1f fps)",
                        writer.frames_written,
                        elapsed,
                        writer.frames_written / max(elapsed, 1e-6),
                    )

            total_frames = writer.frames_written

    wall_clock_elapsed = time.perf_counter() - wall_clock_start
    effective_fps = total_frames / max(wall_clock_elapsed, 1e-6)

    logger.info(
        "Done — %d frames processed in %.1f s (%.1f effective fps). Output: %s",
        total_frames,
        wall_clock_elapsed,
        effective_fps,
        args.output,
    )


if __name__ == "__main__":
    main()
