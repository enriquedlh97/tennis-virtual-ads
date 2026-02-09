#!/usr/bin/env python3
"""Debug script -- run the court calibrator on a single image.

Loads a tennis broadcast frame, runs the TennisCourtDetectorCalibrator,
prints calibration stats, and saves an annotated output image.

Usage
-----
    # Overlay mode (draws projected court lines on the image):
    uv run python scripts/debug_calibrator_image.py \\
        --image_path ../tennis_court_detection/test_images/tennis_pic_01.png \\
        --output_path output_debug.png

    # Keypoints mode (draws detected keypoint circles only):
    uv run python scripts/debug_calibrator_image.py \\
        --image_path ../tennis_court_detection/test_images/tennis_pic_01.png \\
        --output_path output_debug.png \\
        --draw_mode keypoints

    # With custom weights path and resize:
    uv run python scripts/debug_calibrator_image.py \\
        --image_path frame.png \\
        --output_path out.png \\
        --weights_path weights/my_model.pt \\
        --resize 1280x720
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Make the ``src/`` tree importable when running the script directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tennis_virtual_ads.pipeline.calibrators.tennis_court_detector import (  # noqa: E402
    TennisCourtDetectorCalibrator,
)
from tennis_virtual_ads.utils.draw import (  # noqa: E402
    DETECTED_KEYPOINT_COLOR,
    draw_keypoints,
    draw_projected_keypoints,
    draw_projected_lines,
)

logger = logging.getLogger("debug_calibrator_image")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_resize(value: str | None) -> tuple[int, int] | None:
    """Parse ``'WIDTHxHEIGHT'`` into ``(width, height)``."""
    if value is None:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resize format '{value}'. Expected 'WIDTHxHEIGHT'.")
    return int(parts[0]), int(parts[1])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TennisCourtDetector calibrator on a single image.",
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the input tennis broadcast image.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the annotated output image.",
    )
    parser.add_argument(
        "--weights_path",
        default="weights/tennis_court_detector.pt",
        help="Path to model weights (default: weights/tennis_court_detector.pt).",
    )
    parser.add_argument(
        "--resize",
        default=None,
        help="Resize input image to WIDTHxHEIGHT before processing (e.g. '1280x720').",
    )
    parser.add_argument(
        "--draw_mode",
        choices=["keypoints", "overlay"],
        default="overlay",
        help=(
            "Drawing mode: 'overlay' projects court lines via homography; "
            "'keypoints' draws only detected keypoint circles (default: overlay)."
        ),
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

    # --- Load image -------------------------------------------------------
    image = cv2.imread(args.image_path)
    if image is None:
        logger.error("Could not read image: %s", args.image_path)
        sys.exit(1)
    logger.info(
        "Loaded image: %s (%dx%d)",
        args.image_path,
        image.shape[1],
        image.shape[0],
    )

    resize = parse_resize(args.resize)
    if resize is not None:
        image = cv2.resize(image, resize)
        logger.info("Resized to %dx%d", resize[0], resize[1])

    # --- Create calibrator ------------------------------------------------
    calibrator = TennisCourtDetectorCalibrator(
        weights_path=args.weights_path,
    )

    # --- Run calibration --------------------------------------------------
    result = calibrator.estimate(image)

    # --- Print results ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  Calibration Results")
    print("=" * 60)
    print(f"  Confidence        : {result['conf']:.3f}")
    print(f"  H present         : {result['H'] is not None}")
    print(
        f"  Keypoints detected: {result['debug']['detected_keypoint_count']}"
        f" / {result['debug']['total_keypoints']}"
    )
    if result["debug"]["reprojection_error_px"] is not None:
        print(f"  Reproj error (px) : {result['debug']['reprojection_error_px']:.2f}")
    else:
        print("  Reproj error (px) : N/A (no homography)")
    print(f"  Device            : {result['debug']['device']}")
    print(f"  Image size        : {result['debug']['image_size']}")
    print("=" * 60 + "\n")

    # --- Draw output image ------------------------------------------------
    output_image = image.copy()
    homography = result["H"]

    if args.draw_mode == "overlay" and homography is not None:
        draw_projected_lines(output_image, homography, calibrator.court_line_segments)
        draw_projected_keypoints(output_image, homography, calibrator.reference_keypoints)
        if result["keypoints"] is not None:
            draw_keypoints(output_image, result["keypoints"])

    elif args.draw_mode == "overlay" and homography is None:
        logger.warning(
            "Overlay mode requested but no homography was computed. "
            "Falling back to keypoints-only drawing."
        )
        if result["keypoints"] is not None:
            draw_keypoints(output_image, result["keypoints"], color=DETECTED_KEYPOINT_COLOR)

    elif args.draw_mode == "keypoints":
        if result["keypoints"] is not None:
            draw_keypoints(output_image, result["keypoints"])

    # --- Save output ------------------------------------------------------
    cv2.imwrite(args.output_path, output_image)
    logger.info("Saved output image to: %s", args.output_path)
    print(f"Output written to: {args.output_path}")


if __name__ == "__main__":
    main()
