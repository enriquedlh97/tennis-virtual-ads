"""Shared drawing utilities for court overlay and keypoint visualization.

All functions in this module are **generic** -- they accept coordinates and
arrays as parameters rather than importing court-specific data structures.
This keeps the module free of dependencies on any particular calibrator
implementation.

Usage from scripts::

    from tennis_virtual_ads.utils.draw import (
        draw_projected_lines,
        draw_keypoints,
        draw_projected_keypoints,
        COURT_LINE_COLOR,
    )
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour constants (BGR for OpenCV)
# ---------------------------------------------------------------------------

COURT_LINE_COLOR: tuple[int, int, int] = (0, 255, 0)  # Green
DETECTED_KEYPOINT_COLOR: tuple[int, int, int] = (0, 0, 255)  # Red
PROJECTED_KEYPOINT_COLOR: tuple[int, int, int] = (255, 165, 0)  # Orange
STATUS_OK_COLOR: tuple[int, int, int] = (0, 255, 0)  # Green
STATUS_FAIL_COLOR: tuple[int, int, int] = (0, 0, 255)  # Red
TEXT_BG_COLOR: tuple[int, int, int] = (0, 0, 0)  # Black outline


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------


def project_point(homography: np.ndarray, point: tuple[int, int]) -> tuple[int, int] | None:
    """Project a 2D point through a 3x3 homography matrix.

    Parameters
    ----------
    homography : np.ndarray
        3x3 homography matrix.
    point : tuple[int, int]
        Source point ``(x, y)`` in the coordinate system that *homography*
        maps **from**.

    Returns
    -------
    tuple[int, int] | None
        Projected ``(x, y)`` in the target coordinate system, or ``None``
        if the point projects to infinity (``w ~ 0``).
    """
    point_homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float64)
    projected = homography @ point_homogeneous
    if abs(projected[2]) < 1e-10:
        return None
    projected /= projected[2]
    return (round(projected[0]), round(projected[1]))


# ---------------------------------------------------------------------------
# Line drawing
# ---------------------------------------------------------------------------


def draw_projected_lines(
    image: np.ndarray,
    homography: np.ndarray,
    line_segments: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    color: tuple[int, int, int] = COURT_LINE_COLOR,
    thickness: int = 2,
) -> None:
    """Draw line segments projected through *homography* onto *image*.

    Each element of *line_segments* is a pair of points
    ``((x1, y1), (x2, y2))`` in the source coordinate system.  Both
    endpoints are projected through *homography* before drawing.

    Mutates *image* in-place.
    """
    for start_point, end_point in line_segments:
        image_start = project_point(homography, start_point)
        image_end = project_point(homography, end_point)
        if image_start is not None and image_end is not None:
            cv2.line(image, image_start, image_end, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Keypoint drawing
# ---------------------------------------------------------------------------


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int] = DETECTED_KEYPOINT_COLOR,
    radius: int = 6,
    show_index: bool = True,
) -> None:
    """Draw keypoint circles (and optional index labels) on *image*.

    Parameters
    ----------
    keypoints : np.ndarray
        ``(N, 2)`` array of ``(x, y)`` coordinates.  Rows containing
        ``NaN`` are silently skipped.
    show_index : bool
        If ``True``, draw the keypoint index number next to each circle.

    Mutates *image* in-place.
    """
    _frame_height, frame_width = image.shape[:2]
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
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def draw_projected_keypoints(
    image: np.ndarray,
    homography: np.ndarray,
    reference_keypoints: np.ndarray,
    color: tuple[int, int, int] = PROJECTED_KEYPOINT_COLOR,
    radius: int = 4,
) -> None:
    """Project reference keypoints through *homography* and draw them.

    Parameters
    ----------
    reference_keypoints : np.ndarray
        ``(N, 1, 2)`` array of reference points (suitable for
        ``cv2.perspectiveTransform``).

    Mutates *image* in-place.
    """
    projected = cv2.perspectiveTransform(reference_keypoints, homography)
    projected_2d = np.array([np.squeeze(pt) for pt in projected])
    draw_keypoints(image, projected_2d, color=color, radius=radius, show_index=False)


# ---------------------------------------------------------------------------
# Text overlay
# ---------------------------------------------------------------------------


def overlay_text_with_outline(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float,
    foreground_color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw *text* with a black outline for contrast.

    Mutates *image* in-place.
    """
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_BG_COLOR,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        foreground_color,
        thickness,
        cv2.LINE_AA,
    )
