#!/usr/bin/env python3
"""Debug script — run the court calibrator on a single image.

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
import numpy as np

# ---------------------------------------------------------------------------
# Make the ``src/`` tree importable when running the script directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tennis_virtual_ads.pipeline.calibrators._tcd_adapted import (  # noqa: E402
    CourtReference,
    refer_kps,
)
from tennis_virtual_ads.pipeline.calibrators.tennis_court_detector import (  # noqa: E402
    TennisCourtDetectorCalibrator,
)

logger = logging.getLogger("debug_calibrator_image")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

COURT_LINE_COLOR = (0, 255, 0)  # Green — projected court lines
DETECTED_KP_COLOR = (0, 0, 255)  # Red — raw detected keypoints
PROJECTED_KP_COLOR = (255, 165, 0)  # Orange — homography-projected keypoints
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 0, 0)  # Black outline


def project_point(homography: np.ndarray, point: tuple[int, int]) -> tuple[int, int] | None:
    """Project a 2D court-reference point through homography H.

    Returns ``None`` if the projected point is at infinity (w ≈ 0).
    """
    point_homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float64)
    projected = homography @ point_homogeneous
    if abs(projected[2]) < 1e-10:
        return None
    projected /= projected[2]
    return (round(projected[0]), round(projected[1]))


def draw_court_overlay(
    image: np.ndarray,
    homography: np.ndarray,
    court_reference: CourtReference,
    line_thickness: int = 2,
) -> None:
    """Draw all court lines projected through *homography* onto *image*.

    Mutates *image* in-place.
    """
    court_lines = [
        court_reference.baseline_top,
        court_reference.baseline_bottom,
        court_reference.net,
        court_reference.left_court_line,
        court_reference.right_court_line,
        court_reference.left_inner_line,
        court_reference.right_inner_line,
        court_reference.middle_line,
        court_reference.top_inner_line,
        court_reference.bottom_inner_line,
    ]

    for start_point, end_point in court_lines:
        image_start = project_point(homography, start_point)
        image_end = project_point(homography, end_point)
        if image_start is not None and image_end is not None:
            cv2.line(
                image,
                image_start,
                image_end,
                COURT_LINE_COLOR,
                line_thickness,
                cv2.LINE_AA,
            )


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int] = DETECTED_KP_COLOR,
    radius: int = 6,
    show_index: bool = True,
) -> None:
    """Draw keypoint circles (and optional index labels) on *image*.

    *keypoints* is a ``(N, 2)`` array where ``NaN`` rows are skipped.
    Mutates *image* in-place.
    """
    frame_height, frame_width = image.shape[:2]
    font_scale = max(0.35, frame_width / 2500.0)

    for index in range(len(keypoints)):
        x_value, y_value = keypoints[index]
        if np.isnan(x_value) or np.isnan(y_value):
            continue
        centre = (round(x_value), round(y_value))
        cv2.circle(image, centre, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, centre, radius, (0, 0, 0), 1, cv2.LINE_AA)

        if show_index:
            label_position = (centre[0] + radius + 2, centre[1] + 4)
            cv2.putText(
                image,
                str(index),
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                TEXT_BG_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(index),
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )


def draw_projected_keypoints(
    image: np.ndarray,
    homography: np.ndarray,
) -> None:
    """Draw all 14 reference keypoints projected through H onto *image*.

    These are the "corrected" keypoint positions — where the homography
    says each court point should be.  Drawn as orange circles.
    """
    projected = cv2.perspectiveTransform(refer_kps, homography)
    projected_2d = np.array([np.squeeze(pt) for pt in projected])
    draw_keypoints(
        image,
        projected_2d,
        color=PROJECTED_KP_COLOR,
        radius=4,
        show_index=False,
    )


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

    if args.draw_mode == "overlay" and result["H"] is not None:
        # Draw projected court lines
        draw_court_overlay(
            output_image,
            result["H"],
            calibrator.court_reference,
        )
        # Draw projected keypoints (orange) — where H says they should be
        draw_projected_keypoints(output_image, result["H"])
        # Draw raw detected keypoints (red) on top
        if result["keypoints"] is not None:
            draw_keypoints(output_image, result["keypoints"])

    elif args.draw_mode == "overlay" and result["H"] is None:
        logger.warning(
            "Overlay mode requested but no homography was computed. "
            "Falling back to keypoints-only drawing."
        )
        if result["keypoints"] is not None:
            draw_keypoints(output_image, result["keypoints"])

    elif args.draw_mode == "keypoints":
        if result["keypoints"] is not None:
            draw_keypoints(output_image, result["keypoints"])

    # --- Save output ------------------------------------------------------
    cv2.imwrite(args.output_path, output_image)
    logger.info("Saved output image to: %s", args.output_path)
    print(f"Output written to: {args.output_path}")


if __name__ == "__main__":
    main()
