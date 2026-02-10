"""Ad placement specification and court-reference geometry.

This module converts a human-friendly placement spec (anchor name +
width/height/offset ratios) into concrete corner coordinates in the
court reference coordinate system used by ``CourtReference``.

Court reference coordinate system (pixels, from court_reference.py)
-------------------------------------------------------------------
::

    (286, 561)                    (1379, 561)
        +----------------------------+  <- top baseline
        |                            |
        |  (423,1110)    (1242,1110) |  <- top service line
        |       +------------+       |
        |       |            |       |
        |       |    (832)   |       |
        |       |            |       |
   -----+-------+----net-----+-------+----- y=1748
        |       |            |       |
        |       |            |       |
        |       +------------+       |
        |  (423,2386)    (1242,2386) |  <- bottom service line
        |                            |
        +----------------------------+  <- bottom baseline
    (286, 2935)                    (1379, 2935)

The playable court spans x=[286, 1379] and y=[561, 2935].

Ratios are relative to the playable court dimensions:
- ``width_ratio``:  fraction of court width  (1379 - 286 = 1093 px)
- ``height_ratio``: fraction of court height (2935 - 561 = 2374 px)
- ``y_offset_ratio``: how far *inside* the court (toward the net) the
  ad's nearest edge is from the anchor baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Court reference constants (from court_reference.py)
# ---------------------------------------------------------------------------
COURT_LEFT_X = 286
COURT_RIGHT_X = 1379
COURT_TOP_Y = 561  # top baseline
COURT_BOTTOM_Y = 2935  # bottom baseline

COURT_WIDTH = COURT_RIGHT_X - COURT_LEFT_X  # 1093
COURT_HEIGHT = COURT_BOTTOM_Y - COURT_TOP_Y  # 2374
COURT_CENTER_X = (COURT_LEFT_X + COURT_RIGHT_X) / 2.0  # 832.5


# ---------------------------------------------------------------------------
# PlacementSpec
# ---------------------------------------------------------------------------


class PlacementSpec(TypedDict):
    """Human-friendly ad placement specification.

    All ratios are relative to the playable court dimensions.
    """

    anchor: str
    width_ratio: float
    height_ratio: float
    y_offset_ratio: float


# ---------------------------------------------------------------------------
# Prepared placement (compute geometry once, reuse per frame)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedPlacement:
    """A placement spec with precomputed court-reference corners.

    This keeps responsibilities clean:
    - `placement.py` owns court-reference geometry computations.
    - `AdPlacer` only warps pixels given H + precomputed corners.
    """

    placement_spec: PlacementSpec
    court_corners: np.ndarray  # (4, 2) float32, TL/TR/BR/BL in court reference coords


def prepare_placement(placement_spec: PlacementSpec) -> PreparedPlacement:
    """Compute and package court-reference geometry for a placement spec."""
    court_corners = compute_ad_court_corners(placement_spec)
    return PreparedPlacement(
        placement_spec=placement_spec,
        court_corners=court_corners,
    )


def get_precomputed_court_corners(placement: PreparedPlacement | PlacementSpec) -> np.ndarray:
    """Return precomputed corners for use by warping code.

    If you pass a raw `PlacementSpec`, this raises to prevent hidden per-frame
    geometry work inside the warper.
    """
    if isinstance(placement, PreparedPlacement):
        return placement.court_corners
    raise ValueError(
        "Ad placement geometry must be prepared once via prepare_placement(placement_spec)."
    )


# ---------------------------------------------------------------------------
# Anchor definitions
# ---------------------------------------------------------------------------
# Each anchor maps to a (center_x, baseline_y, direction) tuple.
# ``direction`` is -1 for "inward from bottom baseline" (toward net, y
# decreases) and +1 for "inward from top baseline" (toward net, y
# increases).

_ANCHOR_REGISTRY: dict[str, tuple[float, float, int]] = {
    "near_baseline_center": (COURT_CENTER_X, COURT_BOTTOM_Y, -1),
    "near_baseline_top_center": (COURT_CENTER_X, COURT_TOP_Y, 1),
}

AVAILABLE_ANCHORS: list[str] = sorted(_ANCHOR_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ad_court_corners(spec: PlacementSpec) -> np.ndarray:
    """Compute the 4 ad corners in court-reference coordinates.

    Parameters
    ----------
    spec : PlacementSpec
        Placement specification with anchor, width/height/offset ratios.

    Returns
    -------
    np.ndarray
        ``(4, 2)`` float32 array of corners in order:
        top-left, top-right, bottom-right, bottom-left.
        "Top/bottom" here refers to the *ad image* orientation, not
        necessarily the court y-axis direction.

    Raises
    ------
    ValueError
        If the anchor name is not recognised.
    """
    anchor_name = spec["anchor"]
    if anchor_name not in _ANCHOR_REGISTRY:
        available = ", ".join(AVAILABLE_ANCHORS)
        raise ValueError(f"Unknown anchor '{anchor_name}'. Available: {available}")

    center_x, baseline_y, direction = _ANCHOR_REGISTRY[anchor_name]

    ad_width = COURT_WIDTH * spec["width_ratio"]
    ad_height = COURT_HEIGHT * spec["height_ratio"]
    y_offset = COURT_HEIGHT * spec["y_offset_ratio"]

    # The ad's nearest edge to the baseline.
    near_edge_y = baseline_y + direction * y_offset
    # The ad's far edge (away from baseline, toward net).
    far_edge_y = near_edge_y + direction * ad_height

    left_x = center_x - ad_width / 2.0
    right_x = center_x + ad_width / 2.0

    # Order: TL, TR, BR, BL (relative to the ad image).
    # When direction == -1 (bottom baseline): far_edge_y < near_edge_y,
    #   so TL=(left, far_edge), BL=(left, near_edge).
    # When direction == +1 (top baseline):    far_edge_y > near_edge_y,
    #   so TL=(left, near_edge), BL=(left, far_edge).
    if direction == -1:
        # Bottom baseline: ad top is farther from baseline (lower y).
        top_y = far_edge_y
        bottom_y = near_edge_y
    else:
        # Top baseline: ad top is nearer to baseline (lower y).
        top_y = near_edge_y
        bottom_y = far_edge_y

    corners = np.array(
        [
            [left_x, top_y],  # top-left
            [right_x, top_y],  # top-right
            [right_x, bottom_y],  # bottom-right
            [left_x, bottom_y],  # bottom-left
        ],
        dtype=np.float32,
    )
    return corners
