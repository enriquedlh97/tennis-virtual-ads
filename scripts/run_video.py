#!/usr/bin/env python3
"""CLI entrypoint -- read a video, run court calibration, write output.

Usage
-----
    # Passthrough (no calibration):
    python scripts/run_video.py input.mp4 output.mp4

    # With TennisCourtDetector calibrator + court overlay:
    python scripts/run_video.py input.mp4 output.mp4 \\
        --calibrator tennis_court_detector \\
        --draw_mode overlay

    # With temporal keypoint smoothing enabled:
    python scripts/run_video.py input.mp4 output.mp4 \\
        --calibrator tennis_court_detector \\
        --draw_mode overlay \\
        --smooth_keypoints --kp_smooth_alpha 0.7

    # With ad placement (requires an RGBA PNG):
    python scripts/run_video.py input.mp4 output.mp4 \\
        --calibrator tennis_court_detector \\
        --draw_mode overlay --smooth_keypoints \\
        --ad_enable --ad_image_path assets/test_ad.png

    # Quick dev run (every 3rd frame, first 100 frames):
    python scripts/run_video.py input.mp4 output.mp4 \\
        --calibrator tennis_court_detector \\
        --stride 3 --max_frames 100

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

from tennis_virtual_ads.io.video import VideoReader, VideoWriter, reencode_to_h264  # noqa: E402
from tennis_virtual_ads.pipeline.calibrators.base import (  # noqa: E402
    CalibrationResult,
    CourtCalibrator,
)
from tennis_virtual_ads.pipeline.calibrators.dummy import DummyCalibrator  # noqa: E402
from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker  # noqa: E402
from tennis_virtual_ads.pipeline.placer.ad_placer import AdPlacer  # noqa: E402
from tennis_virtual_ads.pipeline.placer.placement import (  # noqa: E402
    AVAILABLE_ANCHORS,
    PlacementSpec,
    prepare_placement,
)
from tennis_virtual_ads.pipeline.temporal.jitter_tracker import JitterTracker  # noqa: E402
from tennis_virtual_ads.pipeline.temporal.keypoint_smoother import KeypointSmoother  # noqa: E402
from tennis_virtual_ads.utils.draw import (  # noqa: E402
    STATUS_FAIL_COLOR,
    STATUS_OK_COLOR,
    draw_keypoints,
    draw_projected_lines,
    overlay_text_with_outline,
)

logger = logging.getLogger("run_video")

DEFAULTS_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"

# Cyan for smoothing HUD, yellow for reset indicator, magenta for ad HUD,
# orange for mask HUD, lime-green for H-stabilizer HUD.
SMOOTH_HUD_COLOR: tuple[int, int, int] = (255, 255, 0)  # Cyan (BGR)
SMOOTH_RESET_COLOR: tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
AD_HUD_COLOR: tuple[int, int, int] = (255, 0, 255)  # Magenta (BGR)
MASK_HUD_COLOR: tuple[int, int, int] = (0, 165, 255)  # Orange (BGR)
HSTAB_HUD_COLOR: tuple[int, int, int] = (0, 255, 128)  # Lime-green (BGR)
HSTAB_HOLD_COLOR: tuple[int, int, int] = (0, 200, 255)  # Amber (BGR)
CUT_HUD_COLOR: tuple[int, int, int] = (0, 0, 255)  # Red (BGR)
BLEND_HUD_COLOR: tuple[int, int, int] = (200, 200, 0)  # Teal (BGR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict."""
    if config_path.exists():
        with open(config_path) as config_file:
            return yaml.safe_load(config_file) or {}
    logger.warning("Config file not found: %s -- using built-in defaults.", config_path)
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


def keypoints_array_to_tuple_list(
    keypoints: np.ndarray,
) -> list[tuple[float | None, float | None]]:
    """Convert a ``(14, 2)`` keypoints array to the tuple-list format.

    NaN entries become ``(None, None)``, which is what
    ``get_trans_matrix`` expects for undetected keypoints.
    """
    result: list[tuple[float | None, float | None]] = []
    for row in keypoints:
        if np.isnan(row[0]) or np.isnan(row[1]):
            result.append((None, None))
        else:
            result.append((float(row[0]), float(row[1])))
    return result


def load_ad_image(image_path: str) -> np.ndarray:
    """Load an RGBA ad image from disk.

    Parameters
    ----------
    image_path : str
        Path to a PNG file with an alpha channel.

    Returns
    -------
    np.ndarray
        ``(H, W, 4)`` uint8 BGRA array.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ValueError
        If the image doesn't have an alpha channel.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Ad image not found: {image_path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read ad image: {image_path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(
            f"Ad image must have an alpha channel (4 channels), "
            f"but got shape {image.shape}. Use a PNG with transparency."
        )
    return image


# ---------------------------------------------------------------------------
# HUD overlay helpers (frame index + calibration status + smoothing status)
# ---------------------------------------------------------------------------


def _compute_font_metrics(frame_width: int) -> tuple[float, int, int]:
    """Return ``(font_scale, thickness, line_height)`` scaled to frame width."""
    font_scale = max(0.5, frame_width / 1280.0)
    thickness = max(1, int(font_scale * 2))
    line_height = int(30 + font_scale * 10)
    return font_scale, thickness, line_height


def overlay_frame_index(frame: np.ndarray, frame_index: int) -> None:
    """Draw the frame index onto the top-left of *frame* (mutates in-place)."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height)
    overlay_text_with_outline(
        frame, f"Frame {frame_index}", position, font_scale, (255, 255, 255), thickness
    )


def overlay_calibration_status(
    frame: np.ndarray,
    result: CalibrationResult,
    is_accepted: bool,
) -> None:
    """Draw calibration status text (line 2 of HUD).  Mutates *frame*."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height + int(font_scale * 30))

    debug = result["debug"]
    detected_count = debug.get("detected_keypoint_count", 0)
    total_keypoints = debug.get("total_keypoints", 14)
    reprojection_error = debug.get("reprojection_error_px")

    if is_accepted:
        error_part = f" err={reprojection_error:.1f}px" if reprojection_error is not None else ""
        text = (
            f"CALIB OK conf={result['conf']:.2f} kp={detected_count}/{total_keypoints}{error_part}"
        )
        color = STATUS_OK_COLOR
    else:
        text = f"NO CALIB conf={result['conf']:.2f} kp={detected_count}/{total_keypoints}"
        color = STATUS_FAIL_COLOR

    overlay_text_with_outline(frame, text, position, font_scale, color, thickness)


def overlay_smoothing_status(
    frame: np.ndarray,
    smoothed_error: float | None,
    did_reset: bool,
) -> None:
    """Draw smoothing status text (line 3 of HUD).  Mutates *frame*."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height + int(font_scale * 60))

    if did_reset:
        text = "SMOOTH=RESET (spike detected)"
        color = SMOOTH_RESET_COLOR
    elif smoothed_error is not None:
        text = f"SMOOTH=ON err={smoothed_error:.1f}px"
        color = SMOOTH_HUD_COLOR
    else:
        text = "SMOOTH=ON (no H)"
        color = SMOOTH_HUD_COLOR

    overlay_text_with_outline(frame, text, position, font_scale, color, thickness)


def overlay_ad_status(
    frame: np.ndarray,
    anchor: str,
    has_smoothing: bool,
    hud_line_offset: int,
) -> None:
    """Draw ad placement status (HUD line 3 or 4).  Mutates *frame*."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height + int(font_scale * 30 * hud_line_offset))

    smooth_label = " (smooth H)" if has_smoothing else ""
    text = f"AD=ON anchor={anchor}{smooth_label}"
    overlay_text_with_outline(frame, text, position, font_scale, AD_HUD_COLOR, thickness)


def overlay_mask_status(
    frame: np.ndarray,
    masker_name: str,
    instance_count: int,
    hud_line_offset: int,
) -> None:
    """Draw occlusion mask status (HUD line).  Mutates *frame*."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height + int(font_scale * 30 * hud_line_offset))

    text = f"MASK={masker_name} persons={instance_count}"
    overlay_text_with_outline(frame, text, position, font_scale, MASK_HUD_COLOR, thickness)


def overlay_mask_debug(
    frame: np.ndarray,
    occlusion_mask: np.ndarray,
) -> None:
    """Draw a small semi-transparent mask preview in the bottom-right corner.

    Renders the occlusion mask as a red-tinted thumbnail (1/4 frame size)
    so the developer can see what the masker is detecting.  Mutates *frame*.
    """
    frame_height, frame_width = frame.shape[:2]
    preview_width = frame_width // 4
    preview_height = frame_height // 4

    # Resize mask to thumbnail size.
    mask_small = cv2.resize(
        occlusion_mask, (preview_width, preview_height), interpolation=cv2.INTER_AREA
    )

    # Create red-tinted overlay: red channel = mask, others = 0.
    overlay = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
    overlay[:, :, 2] = (mask_small * 255).astype(np.uint8)  # Red channel (BGR)

    # Blend into the bottom-right corner of the frame at 70% opacity.
    x_offset = frame_width - preview_width - 10
    y_offset = frame_height - preview_height - 10
    roi = frame[y_offset : y_offset + preview_height, x_offset : x_offset + preview_width]

    blend_alpha = 0.7
    blended = cv2.addWeighted(roi, 1.0 - blend_alpha, overlay, blend_alpha, 0)
    frame[y_offset : y_offset + preview_height, x_offset : x_offset + preview_width] = blended

    # Label it.
    font_scale = max(0.4, frame_width / 2560.0)
    overlay_text_with_outline(
        frame,
        "MASK DEBUG",
        (x_offset + 5, y_offset + 20),
        font_scale,
        (255, 255, 255),
        max(1, int(font_scale * 2)),
    )


def overlay_stabilizer_status(
    frame: np.ndarray,
    alpha: float,
    is_holding: bool,
    hold_count: int,
    max_hold_frames: int,
    did_reject: bool,
    hud_line_offset: int,
) -> None:
    """Draw homography stabilizer status (HUD line).  Mutates *frame*."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height + int(font_scale * 30 * hud_line_offset))

    if is_holding:
        text = f"HSTAB=HOLD ({hold_count}/{max_hold_frames})"
        color = HSTAB_HOLD_COLOR
    elif did_reject:
        text = f"HSTAB=REJECT (outlier) alpha={alpha:.2f}"
        color = HSTAB_HOLD_COLOR
    else:
        text = f"HSTAB=ON alpha={alpha:.2f}"
        color = HSTAB_HUD_COLOR

    overlay_text_with_outline(frame, text, position, font_scale, color, thickness)


def overlay_cut_detected(
    frame: np.ndarray,
    hud_line_offset: int,
) -> None:
    """Draw cut-detected alert (HUD line).  Mutates *frame*."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height + int(font_scale * 30 * hud_line_offset))

    text = "CUT DETECTED -> RESET"
    overlay_text_with_outline(frame, text, position, font_scale, CUT_HUD_COLOR, thickness)


def overlay_blend_status(
    frame: np.ndarray,
    blend_mode: str,
    shade_blur_ksize: int,
    shade_strength: float,
    alpha_feather_px: int,
    hud_line_offset: int,
) -> None:
    """Draw blend mode status (HUD line).  Mutates *frame*."""
    _frame_height, frame_width = frame.shape[:2]
    font_scale, thickness, line_height = _compute_font_metrics(frame_width)
    position = (10, line_height + int(font_scale * 30 * hud_line_offset))

    text = (
        f"BLEND={blend_mode} blur={shade_blur_ksize} "
        f"strength={shade_strength:.1f} feather={alpha_feather_px}"
    )
    overlay_text_with_outline(frame, text, position, font_scale, BLEND_HUD_COLOR, thickness)


def overlay_shade_debug(
    frame: np.ndarray,
    shade_map: np.ndarray,
) -> None:
    """Draw a small shade-map preview in the bottom-left corner.

    Renders the normalised shade map as a grayscale thumbnail so the
    developer can see what illumination the blender is detecting.
    Placed in the bottom-LEFT to avoid overlapping the mask debug
    preview (bottom-right).  Mutates *frame*.
    """
    frame_height, frame_width = frame.shape[:2]
    preview_width = frame_width // 4
    preview_height = frame_height // 4

    # Normalise shade map to 0-255 for visualisation.
    # shade_map is typically in [0.6, 1.4]; map 0.5-1.5 -> 0-255.
    shade_vis = np.clip((shade_map - 0.5) / 1.0, 0.0, 1.0)
    shade_vis = (shade_vis * 255).astype(np.uint8)
    shade_small = cv2.resize(
        shade_vis, (preview_width, preview_height), interpolation=cv2.INTER_AREA
    )

    # Convert to BGR for blending.
    shade_bgr = cv2.cvtColor(shade_small, cv2.COLOR_GRAY2BGR)

    # Blend into the bottom-left corner at 70% opacity.
    x_offset = 10
    y_offset = frame_height - preview_height - 10
    roi = frame[y_offset : y_offset + preview_height, x_offset : x_offset + preview_width]

    blend_alpha = 0.7
    blended = cv2.addWeighted(roi, 1.0 - blend_alpha, shade_bgr, blend_alpha, 0)
    frame[y_offset : y_offset + preview_height, x_offset : x_offset + preview_width] = blended

    # Label it.
    font_scale = max(0.4, frame_width / 2560.0)
    overlay_text_with_outline(
        frame,
        "SHADE DEBUG",
        (x_offset + 5, y_offset + 20),
        font_scale,
        (255, 255, 255),
        max(1, int(font_scale * 2)),
    )


# ---------------------------------------------------------------------------
# Calibrator factory
# ---------------------------------------------------------------------------

CALIBRATOR_NAMES: list[str] = ["dummy", "tennis_court_detector"]


def create_calibrator(name: str, **kwargs: Any) -> CourtCalibrator:
    """Instantiate a calibrator by its registered name."""
    if name == "dummy":
        return DummyCalibrator()

    if name == "tennis_court_detector":
        from tennis_virtual_ads.pipeline.calibrators.tennis_court_detector import (
            TennisCourtDetectorCalibrator,
        )

        weights_path = kwargs.get("weights_path", "weights/tennis_court_detector.pt")
        return TennisCourtDetectorCalibrator(weights_path=weights_path)

    valid_names = ", ".join(sorted(CALIBRATOR_NAMES))
    raise ValueError(f"Unknown calibrator '{name}'. Available: {valid_names}")


# ---------------------------------------------------------------------------
# Masker factory
# ---------------------------------------------------------------------------

MASKER_NAMES: list[str] = ["none", "person"]


def create_masker(name: str, **kwargs: Any) -> OcclusionMasker:
    """Instantiate an occlusion masker by its registered name.

    Uses lazy imports so that ``torch`` / ``torchvision`` are only loaded
    when the ``person`` masker is actually requested.
    """
    if name == "person":
        from tennis_virtual_ads.pipeline.maskers.person_masker import PersonMasker

        return PersonMasker(
            confidence_threshold=kwargs.get("confidence_threshold", 0.5),
            device=kwargs.get("device"),
        )

    valid_names = ", ".join(sorted(MASKER_NAMES))
    raise ValueError(f"Unknown masker '{name}'. Available: {valid_names}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Tennis Virtual Ads -- process a video with court calibration "
            "and optional overlay drawing."
        ),
    )
    parser.add_argument("input", help="Path to input video file (e.g. match.mp4)")
    parser.add_argument("output", help="Path to output video file (e.g. output.mp4)")

    # --- Frame selection --------------------------------------------------
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

    # --- Calibration ------------------------------------------------------
    parser.add_argument(
        "--calibrator",
        type=str,
        default=None,
        choices=CALIBRATOR_NAMES,
        help="Court calibrator to use (default: none)",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to calibrator model weights (default: weights/tennis_court_detector.pt)",
    )
    parser.add_argument(
        "--calib_conf_threshold",
        type=float,
        default=0.30,
        help="Minimum confidence to accept a calibration result (default: 0.30)",
    )
    parser.add_argument(
        "--draw_mode",
        type=str,
        choices=["overlay", "keypoints", "none"],
        default="overlay",
        help=(
            "Drawing mode: 'overlay' projects court lines; 'keypoints' draws "
            "detected keypoints; 'none' shows only status text (default: overlay)"
        ),
    )

    # --- Temporal smoothing -----------------------------------------------
    parser.add_argument(
        "--smooth_keypoints",
        action="store_true",
        default=False,
        help="Enable EMA temporal smoothing of keypoints to reduce jitter.",
    )
    parser.add_argument(
        "--kp_smooth_alpha",
        type=float,
        default=0.7,
        help="EMA blending factor: higher = more responsive, lower = smoother (default: 0.7)",
    )
    parser.add_argument(
        "--reset_on_err_spike",
        action="store_true",
        default=True,
        help="Reset smoother when reprojection error spikes (default: true).",
    )
    parser.add_argument(
        "--no_reset_on_err_spike",
        action="store_false",
        dest="reset_on_err_spike",
        help="Disable automatic smoother reset on error spikes.",
    )
    parser.add_argument(
        "--err_spike_factor",
        type=float,
        default=2.0,
        help="Reset if error > factor * median(recent errors) (default: 2.0)",
    )

    # --- Homography stabilization -----------------------------------------
    parser.add_argument(
        "--stabilize_h",
        action="store_true",
        default=False,
        help="Enable EMA temporal stabilization of the homography matrix.",
    )
    parser.add_argument(
        "--h_alpha",
        type=float,
        default=0.9,
        help=(
            "EMA blending factor for H-space: higher = smoother / slower to react "
            "(default: 0.9). Alpha weights the history; (1-alpha) weights the new observation."
        ),
    )
    parser.add_argument(
        "--hold_frames",
        type=int,
        default=15,
        help="Max frames to hold last-good H when calibration fails (default: 15).",
    )
    parser.add_argument(
        "--h_spike_factor",
        type=float,
        default=2.0,
        help=(
            "Outlier rejection threshold for H-stabilizer: reject when error or "
            "projected-point displacement > factor * median(recent) (default: 2.0)."
        ),
    )

    # --- Scene-cut detection -----------------------------------------------
    parser.add_argument(
        "--cut_detection",
        action="store_true",
        default=True,
        help="Enable scene-cut detection to reset temporal state on camera changes (default: true).",
    )
    parser.add_argument(
        "--no_cut_detection",
        action="store_false",
        dest="cut_detection",
        help="Disable scene-cut detection.",
    )
    parser.add_argument(
        "--cut_frame_diff_thresh",
        type=float,
        default=18.0,
        help="Mean absolute frame diff threshold for cut detection (default: 18.0).",
    )
    parser.add_argument(
        "--cut_proj_jump_thresh",
        type=float,
        default=40.0,
        help="Mean projected-point displacement threshold in pixels (default: 40.0).",
    )
    parser.add_argument(
        "--cut_cooldown_frames",
        type=int,
        default=10,
        help="Frames to suppress cut triggers after a confirmed cut (default: 10).",
    )

    # --- Ad placement -----------------------------------------------------
    parser.add_argument(
        "--ad_enable",
        action="store_true",
        default=False,
        help="Enable ad placement on the court surface.",
    )
    parser.add_argument(
        "--ad_image_path",
        type=str,
        default=None,
        help="Path to RGBA PNG ad image (overrides config).",
    )
    parser.add_argument(
        "--ad_anchor",
        type=str,
        default=None,
        choices=AVAILABLE_ANCHORS,
        help=f"Anchor position on court (default: near_baseline_center). Choices: {AVAILABLE_ANCHORS}",
    )
    parser.add_argument(
        "--ad_width_ratio",
        type=float,
        default=None,
        help="Ad width as fraction of court width (default: 0.35).",
    )
    parser.add_argument(
        "--ad_height_ratio",
        type=float,
        default=None,
        help="Ad height as fraction of court height (default: 0.12).",
    )
    parser.add_argument(
        "--ad_y_offset_ratio",
        type=float,
        default=None,
        help="Offset from baseline toward net as fraction of court height (default: 0.06).",
    )

    # --- Occlusion masking ------------------------------------------------
    parser.add_argument(
        "--masker",
        type=str,
        default="none",
        choices=MASKER_NAMES,
        help="Occlusion masker to use (default: none).",
    )
    parser.add_argument(
        "--masker_conf_threshold",
        type=float,
        default=0.5,
        help="Minimum detection score to include an instance in the mask (default: 0.5).",
    )
    parser.add_argument(
        "--mask_dilate_px",
        type=int,
        default=3,
        help="Pixels to dilate the occlusion mask by (covers rackets near body; default: 3).",
    )
    parser.add_argument(
        "--mask_debug",
        action="store_true",
        default=False,
        help="Show a mask preview overlay in the bottom-right corner of the output.",
    )

    # --- Compositing / blend mode -----------------------------------------
    parser.add_argument(
        "--blend_mode",
        type=str,
        default="naive",
        choices=["naive", "painted_v1"],
        help=(
            "Ad compositing mode: 'naive' = flat alpha blend (default); "
            "'painted_v1' = shadow-preserving blend that inherits court illumination."
        ),
    )
    parser.add_argument(
        "--shade_blur_ksize",
        type=int,
        default=41,
        help="Gaussian blur kernel size for illumination extraction (must be odd; default: 41).",
    )
    parser.add_argument(
        "--shade_strength",
        type=float,
        default=1.0,
        help="Shade exponent: 1.0 = natural, >1.0 = exaggerated shadow effect (default: 1.0).",
    )
    parser.add_argument(
        "--alpha_feather_px",
        type=int,
        default=3,
        help="Gaussian blur radius for alpha edge softening (0 = off; default: 3).",
    )
    parser.add_argument(
        "--blend_debug",
        action="store_true",
        default=False,
        help="Show a shade-map preview overlay in the bottom-left corner of the output.",
    )

    # --- Jitter tracking --------------------------------------------------
    parser.add_argument(
        "--no_jitter_tracker",
        action="store_true",
        default=False,
        help="Disable jitter tracking (default: enabled when calibration is active).",
    )

    # --- Config -----------------------------------------------------------
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

    draw_mode: str = args.draw_mode
    calib_conf_threshold: float = args.calib_conf_threshold
    smooth_enabled: bool = args.smooth_keypoints

    # --- Calibrator --------------------------------------------------------
    calibrator_name: str | None = (
        args.calibrator if args.calibrator is not None else config.get("calibrator", None)
    )
    calibrator: CourtCalibrator | None = None
    calibrator_court_lines: list[tuple[tuple[int, int], tuple[int, int]]] | None = None

    if calibrator_name is not None:
        calibrator_kwargs: dict[str, Any] = {}
        if args.weights_path is not None:
            calibrator_kwargs["weights_path"] = args.weights_path
        calibrator = create_calibrator(calibrator_name, **calibrator_kwargs)
        logger.info("Using calibrator: %s", calibrator_name)

        if hasattr(calibrator, "court_line_segments"):
            calibrator_court_lines = calibrator.court_line_segments

    # --- Keypoint smoother (optional) -------------------------------------
    smoother: KeypointSmoother | None = None
    # get_trans_matrix and _compute_reprojection_error are needed to
    # recompute H from smoothed keypoints. Lazy-imported below.
    recompute_homography = None
    compute_reproj_error = None

    if smooth_enabled and calibrator is not None:
        smoother = KeypointSmoother(
            alpha=args.kp_smooth_alpha,
            enable_spike_reset=args.reset_on_err_spike,
            spike_factor=args.err_spike_factor,
        )
        # Import homography tools (already loaded as part of the calibrator).
        from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.homography import (
            get_trans_matrix,
            refer_kps,
        )

        recompute_homography = get_trans_matrix
        _refer_kps = refer_kps

        def _compute_smooth_reproj_error(
            kps_tuples: list[tuple[float | None, float | None]],
            homography: np.ndarray,
        ) -> float | None:
            """Mean reprojection error for smoothed keypoints."""
            import cv2

            projected = cv2.perspectiveTransform(_refer_kps, homography)
            errors: list[float] = []
            for idx in range(14):
                if kps_tuples[idx][0] is not None:
                    detected = np.array(kps_tuples[idx])
                    proj_pt = projected[idx].flatten()
                    errors.append(float(np.linalg.norm(detected - proj_pt)))
            return float(np.mean(errors)) if errors else None

        compute_reproj_error = _compute_smooth_reproj_error

        logger.info(
            "Keypoint smoothing enabled: alpha=%.2f  spike_reset=%s  spike_factor=%.1f",
            args.kp_smooth_alpha,
            args.reset_on_err_spike,
            args.err_spike_factor,
        )

    # --- Homography stabilizer (optional) ---------------------------------
    stabilize_h_enabled: bool = args.stabilize_h
    homography_stabilizer = None
    h_alpha: float = args.h_alpha

    if stabilize_h_enabled and calibrator is not None:
        from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.homography import (
            refer_kps as _stab_refer_kps,
        )
        from tennis_virtual_ads.pipeline.temporal.homography_stabilizer import (
            HomographyStabilizer,
        )

        homography_stabilizer = HomographyStabilizer(
            reference_points=_stab_refer_kps.copy(),
            alpha=args.h_alpha,
            max_hold_frames=args.hold_frames,
            spike_factor=args.h_spike_factor,
        )
        logger.info(
            "Homography stabilizer enabled: alpha=%.2f  hold_frames=%d  spike_factor=%.1f",
            args.h_alpha,
            args.hold_frames,
            args.h_spike_factor,
        )

    # --- Jitter trackers (optional) ---------------------------------------
    # We keep up to three trackers: raw, smoothed (keypoint EMA),
    # and stabilized (H-space EMA).  This lets us quantify the
    # improvement each stage provides.
    jitter_tracker_enabled = not args.no_jitter_tracker and calibrator is not None
    raw_jitter_tracker: JitterTracker | None = None
    smoothed_jitter_tracker: JitterTracker | None = None
    stabilized_jitter_tracker: JitterTracker | None = None

    if jitter_tracker_enabled:
        from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.homography import (
            refer_kps as _jitter_refer_kps,
        )

        raw_jitter_tracker = JitterTracker(reference_points=_jitter_refer_kps.copy())
        if smooth_enabled:
            smoothed_jitter_tracker = JitterTracker(reference_points=_jitter_refer_kps.copy())
        if stabilize_h_enabled:
            stabilized_jitter_tracker = JitterTracker(reference_points=_jitter_refer_kps.copy())
        jitter_labels = ["raw"]
        if smooth_enabled:
            jitter_labels.append("smoothed")
        if stabilize_h_enabled:
            jitter_labels.append("stabilized")
        logger.info("Jitter tracking enabled (%s)", " + ".join(jitter_labels))

    # --- Scene-cut detector (optional) ------------------------------------
    cut_detection_enabled: bool = args.cut_detection and calibrator is not None
    cut_detector = None

    if cut_detection_enabled:
        from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.homography import (
            refer_kps as _cut_refer_kps,
        )
        from tennis_virtual_ads.pipeline.temporal.cut_detector import CutDetector

        cut_detector = CutDetector(
            reference_points=_cut_refer_kps.copy(),
            frame_diff_threshold=args.cut_frame_diff_thresh,
            projection_jump_threshold=args.cut_proj_jump_thresh,
            cooldown_frames=args.cut_cooldown_frames,
        )
        logger.info(
            "Cut detection enabled: frame_diff_thresh=%.1f  proj_jump_thresh=%.1f  "
            "cooldown=%d frames",
            args.cut_frame_diff_thresh,
            args.cut_proj_jump_thresh,
            args.cut_cooldown_frames,
        )

    # --- Ad placement (optional) ------------------------------------------
    ad_config: dict[str, Any] = config.get("ad", {})
    ad_enabled: bool = args.ad_enable or ad_config.get("enabled", False)
    ad_placer: AdPlacer | None = None
    ad_rgba: np.ndarray | None = None
    prepared_ad_placement = None
    ad_anchor_name: str = ""

    if ad_enabled:
        ad_image_path = args.ad_image_path or ad_config.get("image_path")
        if ad_image_path is None:
            logger.error(
                "Ad placement enabled but no image path provided. "
                "Use --ad_image_path or set ad.image_path in config."
            )
            sys.exit(1)

        ad_rgba = load_ad_image(ad_image_path)
        logger.info(
            "Loaded ad image: %s  (%dx%d)", ad_image_path, ad_rgba.shape[1], ad_rgba.shape[0]
        )

        ad_anchor_name = args.ad_anchor or ad_config.get("anchor", "near_baseline_center")
        placement_spec = PlacementSpec(
            anchor=ad_anchor_name,
            width_ratio=args.ad_width_ratio
            if args.ad_width_ratio is not None
            else ad_config.get("width_ratio", 0.35),
            height_ratio=args.ad_height_ratio
            if args.ad_height_ratio is not None
            else ad_config.get("height_ratio", 0.12),
            y_offset_ratio=args.ad_y_offset_ratio
            if args.ad_y_offset_ratio is not None
            else ad_config.get("y_offset_ratio", 0.06),
        )
        prepared_ad_placement = prepare_placement(placement_spec)
        ad_placer = AdPlacer()
        logger.info(
            "Ad placement: anchor=%s  width_ratio=%.2f  height_ratio=%.2f  y_offset=%.2f",
            placement_spec["anchor"],
            placement_spec["width_ratio"],
            placement_spec["height_ratio"],
            placement_spec["y_offset_ratio"],
        )

    # --- Occlusion masker (optional) --------------------------------------
    masker_name: str = args.masker
    masker: OcclusionMasker | None = None
    mask_dilate_px: int = args.mask_dilate_px
    mask_debug: bool = args.mask_debug
    masker_enabled: bool = masker_name != "none"

    if masker_enabled:
        masker = create_masker(
            masker_name,
            confidence_threshold=args.masker_conf_threshold,
        )
        logger.info(
            "Occlusion masker: %s  conf_threshold=%.2f  dilate_px=%d  debug=%s",
            masker_name,
            args.masker_conf_threshold,
            mask_dilate_px,
            mask_debug,
        )

    # --- Blend mode (painted compositing, optional) -----------------------
    blend_mode: str = args.blend_mode
    blend_debug: bool = args.blend_debug
    _painted_composite = None  # lazy-loaded below

    # Outer-court corners in court-reference coordinates.  Used to compute
    # a per-frame court mask for shade normalisation.
    _outer_court_corners_ref: np.ndarray | None = None

    if blend_mode == "painted_v1":
        from tennis_virtual_ads.pipeline.compositor.painted_blend import (
            painted_composite as _painted_composite_fn,
        )

        _painted_composite = _painted_composite_fn

        from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.court_reference import (
            CourtReference,
        )

        _court_ref = CourtReference()
        # border_points = [baseline_top_left, baseline_top_right,
        #                   baseline_bottom_right, baseline_bottom_left]
        _outer_court_corners_ref = np.array(_court_ref.border_points, dtype=np.float32).reshape(
            -1, 1, 2
        )

        logger.info(
            "Painted blend enabled: blur=%d  strength=%.1f  feather=%d  debug=%s",
            args.shade_blur_ksize,
            args.shade_strength,
            args.alpha_feather_px,
            blend_debug,
        )

    def _compute_court_mask(
        homography: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        """Project outer court corners and fill a polygon mask."""
        if _outer_court_corners_ref is None:
            return None
        image_corners = cv2.perspectiveTransform(_outer_court_corners_ref, homography)
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = image_corners.reshape(-1, 2).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        return mask

    logger.info(
        "Settings -- start_frame=%d  max_frames=%s  stride=%d  resize=%s  "
        "calibrator=%s  draw_mode=%s  conf_threshold=%.2f  smooth=%s  "
        "stabilize_h=%s  cut_detect=%s  jitter_track=%s  ad=%s  masker=%s  blend=%s",
        start_frame,
        max_frames,
        stride,
        resize,
        calibrator_name,
        draw_mode,
        calib_conf_threshold,
        smooth_enabled,
        stabilize_h_enabled,
        cut_detection_enabled,
        jitter_tracker_enabled,
        ad_enabled,
        masker_name,
        blend_mode,
    )

    # --- Process video ----------------------------------------------------
    wall_clock_start = time.perf_counter()
    accepted_count = 0
    rejected_count = 0
    reset_count = 0
    cut_count = 0

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
                # --- HUD: frame index ------------------------------------
                overlay_frame_index(frame, frame_index)

                # --- Calibration -----------------------------------------
                if calibrator is not None:
                    calibration_result = calibrator.estimate(frame)
                    raw_homography = calibration_result["H"]
                    confidence = calibration_result["conf"]
                    raw_keypoints = calibration_result["keypoints"]
                    raw_error = calibration_result["debug"].get("reprojection_error_px")

                    is_accepted = raw_homography is not None and confidence >= calib_conf_threshold

                    # --- Scene-cut detection (before temporal processing) --
                    is_cut = False
                    if cut_detector is not None:
                        is_cut = cut_detector.update(frame, raw_homography, confidence, raw_error)
                        if is_cut:
                            cut_count += 1
                            logger.info(
                                "Cut detected at frame %d  (frame_diff=%.1f  proj_jump=%s)",
                                frame_index,
                                cut_detector.last_frame_diff or 0.0,
                                (
                                    f"{cut_detector.last_projection_jump:.1f}"
                                    if cut_detector.last_projection_jump is not None
                                    else "N/A"
                                ),
                            )
                            # Reset all temporal state.
                            if smoother is not None:
                                smoother.reset()
                            if homography_stabilizer is not None:
                                homography_stabilizer.reset()
                            if raw_jitter_tracker is not None:
                                raw_jitter_tracker.reset()
                            if smoothed_jitter_tracker is not None:
                                smoothed_jitter_tracker.reset()
                            if stabilized_jitter_tracker is not None:
                                stabilized_jitter_tracker.reset()

                    # --- Jitter tracking: raw H --------------------------
                    if (
                        raw_jitter_tracker is not None
                        and is_accepted
                        and raw_homography is not None
                    ):
                        raw_jitter_tracker.update(raw_homography)

                    # --- Smoothing (optional) ----------------------------
                    homography_for_drawing = raw_homography
                    smoothed_error: float | None = None
                    did_reset = False

                    if (
                        smoother is not None
                        and is_accepted
                        and raw_keypoints is not None
                        and recompute_homography is not None
                        and compute_reproj_error is not None
                    ):
                        smoothed_kps = smoother.update(raw_keypoints, raw_error)
                        did_reset = smoother.did_reset_this_frame

                        if did_reset:
                            reset_count += 1

                        # Recompute H from smoothed keypoints.
                        smoothed_tuples = keypoints_array_to_tuple_list(smoothed_kps)
                        smoothed_h = recompute_homography(smoothed_tuples)

                        if smoothed_h is not None:
                            homography_for_drawing = smoothed_h
                            smoothed_error = compute_reproj_error(smoothed_tuples, smoothed_h)

                            # --- Jitter tracking: smoothed H -------------
                            if smoothed_jitter_tracker is not None:
                                smoothed_jitter_tracker.update(smoothed_h)

                    if is_accepted:
                        accepted_count += 1
                    else:
                        rejected_count += 1

                    # --- Homography stabilization (optional) --------------
                    # Called on EVERY frame (even rejected ones) so the
                    # hold-last-good counter advances correctly.
                    stabilizer_is_holding = False
                    stabilizer_did_reject = False

                    if homography_stabilizer is not None:
                        # Determine which error to pass: smoothed if
                        # keypoint smoothing is active, else raw.
                        stabilizer_input_error = (
                            smoothed_error if smoother is not None else raw_error
                        )

                        if is_accepted and homography_for_drawing is not None:
                            H_stable = homography_stabilizer.update(
                                homography_for_drawing,
                                confidence,
                                stabilizer_input_error,
                            )
                        else:
                            # Calibration failed/rejected: trigger hold.
                            H_stable = homography_stabilizer.update(None, 0.0, None)

                        stabilizer_is_holding = homography_stabilizer.is_holding
                        stabilizer_did_reject = homography_stabilizer.did_reject_this_frame

                        if H_stable is not None:
                            homography_for_drawing = H_stable

                            # Jitter tracking: stabilized H.
                            if stabilized_jitter_tracker is not None:
                                stabilized_jitter_tracker.update(H_stable)

                    # --- Determine if we have a usable H ------------------
                    # When the stabilizer is active, it may provide a held
                    # H even when this frame's calibration was rejected.
                    has_usable_homography = homography_for_drawing is not None and (
                        is_accepted or stabilizer_is_holding
                    )

                    # --- HUD: calibration status -------------------------
                    overlay_calibration_status(frame, calibration_result, is_accepted)

                    # --- HUD: smoothing status (line 3) ------------------
                    if smoother is not None:
                        overlay_smoothing_status(frame, smoothed_error, did_reset)

                    # --- Drawing -----------------------------------------
                    if (
                        draw_mode == "overlay"
                        and has_usable_homography
                        and homography_for_drawing is not None
                        and calibrator_court_lines is not None
                    ):
                        draw_projected_lines(frame, homography_for_drawing, calibrator_court_lines)

                    if draw_mode == "keypoints" and raw_keypoints is not None:
                        draw_keypoints(frame, raw_keypoints, show_index=False)

                    # --- Occlusion masking --------------------------------
                    occlusion_mask: np.ndarray | None = None
                    mask_instance_count: int = 0

                    if masker is not None:
                        masker_result = masker.mask(frame)
                        occlusion_mask = masker_result["mask"]
                        mask_instance_count = masker_result["debug"].get("instance_count", 0)

                        # Dilate the mask slightly to cover rackets and
                        # limbs near the body edge that the model may miss.
                        if mask_dilate_px > 0 and np.any(occlusion_mask > 0):
                            dilate_kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (2 * mask_dilate_px + 1, 2 * mask_dilate_px + 1),
                            )
                            occlusion_mask = cv2.dilate(occlusion_mask, dilate_kernel, iterations=1)

                    # --- Ad placement ------------------------------------
                    blend_debug_payload: dict[str, Any] = {}

                    if (
                        ad_placer is not None
                        and ad_rgba is not None
                        and prepared_ad_placement is not None
                        and has_usable_homography
                        and homography_for_drawing is not None
                    ):
                        warped_rgba, warped_mask = ad_placer.warp(
                            ad_rgba,
                            homography_for_drawing,
                            prepared_ad_placement,
                            frame.shape,
                        )

                        # When an occlusion mask is active, punch through
                        # the ad alpha so that players remain visible.
                        if occlusion_mask is not None:
                            effective_alpha = warped_mask * (1.0 - occlusion_mask)
                        else:
                            effective_alpha = warped_mask

                        # --- Compositing (naive or painted) ---------------
                        if blend_mode == "painted_v1":
                            assert _painted_composite is not None
                            # Compute court mask from 4 outer corners.
                            court_mask = _compute_court_mask(homography_for_drawing, frame.shape)
                            blend_debug_payload = _painted_composite(
                                frame,
                                warped_rgba,
                                effective_alpha,
                                court_mask,
                                shade_blur_ksize=args.shade_blur_ksize,
                                shade_strength=args.shade_strength,
                                alpha_feather_px=args.alpha_feather_px,
                            )
                        else:
                            ad_placer.composite(frame, warped_rgba, effective_alpha)

                    # --- HUD: ad / stabilizer / mask / blend status -------
                    # Count how many HUD lines are already used so each
                    # status line lands on the correct row.
                    next_hud_line = 2
                    if smoother is not None:
                        next_hud_line = 3

                    if homography_stabilizer is not None:
                        overlay_stabilizer_status(
                            frame,
                            h_alpha,
                            stabilizer_is_holding,
                            homography_stabilizer.hold_count,
                            args.hold_frames,
                            stabilizer_did_reject,
                            next_hud_line,
                        )
                        next_hud_line += 1

                    if ad_placer is not None:
                        overlay_ad_status(frame, ad_anchor_name, smooth_enabled, next_hud_line)
                        next_hud_line += 1

                    # --- HUD: mask status --------------------------------
                    if masker is not None:
                        overlay_mask_status(frame, masker_name, mask_instance_count, next_hud_line)
                        next_hud_line += 1

                    # --- HUD: blend status --------------------------------
                    if blend_mode != "naive" and ad_placer is not None:
                        overlay_blend_status(
                            frame,
                            blend_mode,
                            args.shade_blur_ksize,
                            args.shade_strength,
                            args.alpha_feather_px,
                            next_hud_line,
                        )
                        next_hud_line += 1

                    # --- HUD: cut detected --------------------------------
                    if is_cut:
                        overlay_cut_detected(frame, next_hud_line)

                    # --- Debug: mask overlay (bottom-right) ---------------
                    if mask_debug and occlusion_mask is not None:
                        overlay_mask_debug(frame, occlusion_mask)

                    # --- Debug: shade overlay (bottom-left) ---------------
                    if blend_debug and blend_debug_payload and "shade_map" in blend_debug_payload:
                        overlay_shade_debug(frame, blend_debug_payload["shade_map"])

                writer.write(frame)

                if writer.frames_written % 100 == 0:
                    elapsed = time.perf_counter() - wall_clock_start
                    logger.info(
                        "Progress: %d frames written  (%.1f s elapsed, ~%.1f fps)  "
                        "calib_ok=%d  calib_fail=%d  resets=%d",
                        writer.frames_written,
                        elapsed,
                        writer.frames_written / max(elapsed, 1e-6),
                        accepted_count,
                        rejected_count,
                        reset_count,
                    )

            total_frames = writer.frames_written

    wall_clock_elapsed = time.perf_counter() - wall_clock_start
    effective_fps = total_frames / max(wall_clock_elapsed, 1e-6)

    logger.info(
        "Done -- %d frames in %.1f s (%.1f fps). Output: %s",
        total_frames,
        wall_clock_elapsed,
        effective_fps,
        args.output,
    )

    # --- Re-encode to H.264 for broad playback compatibility --------------
    reencode_to_h264(args.output)

    if calibrator is not None:
        logger.info(
            "Calibration stats -- accepted=%d  rejected=%d  accept_rate=%.1f%%  resets=%d  cuts=%d",
            accepted_count,
            rejected_count,
            100.0 * accepted_count / max(accepted_count + rejected_count, 1),
            reset_count,
            cut_count,
        )

    # --- Jitter summary ---------------------------------------------------
    if raw_jitter_tracker is not None:
        raw_summary = raw_jitter_tracker.get_summary()
        if raw_summary is not None:
            logger.info("Jitter (raw H)      -- %s", raw_summary.to_log_string())
        else:
            logger.info("Jitter (raw H)      -- not enough frames to compute")

    if smoothed_jitter_tracker is not None:
        smooth_summary = smoothed_jitter_tracker.get_summary()
        if smooth_summary is not None:
            logger.info("Jitter (smoothed H) -- %s", smooth_summary.to_log_string())
        else:
            logger.info("Jitter (smoothed H) -- not enough frames to compute")

    if stabilized_jitter_tracker is not None:
        stab_summary = stabilized_jitter_tracker.get_summary()
        if stab_summary is not None:
            logger.info("Jitter (stabilized H) -- %s", stab_summary.to_log_string())
        else:
            logger.info("Jitter (stabilized H) -- not enough frames to compute")


if __name__ == "__main__":
    main()
