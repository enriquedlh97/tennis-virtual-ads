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

# Cyan for smoothing HUD, yellow for reset indicator, magenta for ad HUD.
SMOOTH_HUD_COLOR: tuple[int, int, int] = (255, 255, 0)  # Cyan (BGR)
SMOOTH_RESET_COLOR: tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
AD_HUD_COLOR: tuple[int, int, int] = (255, 0, 255)  # Magenta (BGR)


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

    # --- Jitter trackers (optional) ---------------------------------------
    # We keep two: one for the raw homography and one for the smoothed
    # homography (when smoothing is enabled).  This lets us quantify the
    # improvement directly.
    jitter_tracker_enabled = not args.no_jitter_tracker and calibrator is not None
    raw_jitter_tracker: JitterTracker | None = None
    smoothed_jitter_tracker: JitterTracker | None = None

    if jitter_tracker_enabled:
        from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.homography import (
            refer_kps as _jitter_refer_kps,
        )

        raw_jitter_tracker = JitterTracker(reference_points=_jitter_refer_kps.copy())
        if smooth_enabled:
            smoothed_jitter_tracker = JitterTracker(reference_points=_jitter_refer_kps.copy())
        logger.info("Jitter tracking enabled (raw%s)", " + smoothed" if smooth_enabled else "")

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

    logger.info(
        "Settings -- start_frame=%d  max_frames=%s  stride=%d  resize=%s  "
        "calibrator=%s  draw_mode=%s  conf_threshold=%.2f  smooth=%s  "
        "jitter_track=%s  ad=%s",
        start_frame,
        max_frames,
        stride,
        resize,
        calibrator_name,
        draw_mode,
        calib_conf_threshold,
        smooth_enabled,
        jitter_tracker_enabled,
        ad_enabled,
    )

    # --- Process video ----------------------------------------------------
    wall_clock_start = time.perf_counter()
    accepted_count = 0
    rejected_count = 0
    reset_count = 0

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

                    # --- HUD: calibration status -------------------------
                    overlay_calibration_status(frame, calibration_result, is_accepted)

                    # --- HUD: smoothing status (line 3) ------------------
                    if smoother is not None:
                        overlay_smoothing_status(frame, smoothed_error, did_reset)

                    # --- Drawing -----------------------------------------
                    if (
                        draw_mode == "overlay"
                        and is_accepted
                        and homography_for_drawing is not None
                        and calibrator_court_lines is not None
                    ):
                        draw_projected_lines(frame, homography_for_drawing, calibrator_court_lines)

                    if draw_mode == "keypoints" and raw_keypoints is not None:
                        draw_keypoints(frame, raw_keypoints, show_index=False)

                    # --- Ad placement ------------------------------------
                    if (
                        ad_placer is not None
                        and ad_rgba is not None
                        and prepared_ad_placement is not None
                        and is_accepted
                        and homography_for_drawing is not None
                    ):
                        warped_rgba, warped_mask = ad_placer.warp(
                            ad_rgba,
                            homography_for_drawing,
                            prepared_ad_placement,
                            frame.shape,
                        )
                        ad_placer.composite(frame, warped_rgba, warped_mask)

                    # --- HUD: ad status ----------------------------------
                    if ad_placer is not None:
                        # Ad HUD goes on the next available line after
                        # smoothing (line 3) or calibration (line 2).
                        hud_line = 3 if smoother is not None else 2
                        overlay_ad_status(frame, ad_anchor_name, smooth_enabled, hud_line)

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
            "Calibration stats -- accepted=%d  rejected=%d  accept_rate=%.1f%%  resets=%d",
            accepted_count,
            rejected_count,
            100.0 * accepted_count / max(accepted_count + rejected_count, 1),
            reset_count,
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


if __name__ == "__main__":
    main()
