"""Court reference model — adapted from TennisCourtDetector.

Original source: https://github.com/yastrebksv/TennisCourtDetector
File: court_reference.py

Adaptations:
- Removed unused ``import matplotlib.pyplot as plt``.
- No functional changes.

The court coordinate system uses pixel units on a reference image where
the full court (including margins) is approximately 1665 x 3496 pixels.
14 keypoints are defined at line intersections and line endpoints.
"""

import cv2
import numpy as np


class CourtReference:
    """Court reference model with 14 keypoints and 12 homography configurations."""

    def __init__(self) -> None:
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)

        self.key_points = [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
            *self.middle_line,
        ]

        self.border_points = [*self.baseline_top, *self.baseline_bottom[::-1]]

        self.court_conf = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [
                self.left_inner_line[0],
                self.right_inner_line[0],
                self.left_inner_line[1],
                self.right_inner_line[1],
            ],
            3: [
                self.left_inner_line[0],
                self.right_court_line[0],
                self.left_inner_line[1],
                self.right_court_line[1],
            ],
            4: [
                self.left_court_line[0],
                self.right_inner_line[0],
                self.left_court_line[1],
                self.right_inner_line[1],
            ],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [
                *self.top_inner_line,
                self.left_inner_line[1],
                self.right_inner_line[1],
            ],
            7: [
                self.left_inner_line[0],
                self.right_inner_line[0],
                *self.bottom_inner_line,
            ],
            8: [
                self.right_inner_line[0],
                self.right_court_line[0],
                self.right_inner_line[1],
                self.right_court_line[1],
            ],
            9: [
                self.left_court_line[0],
                self.left_inner_line[0],
                self.left_court_line[1],
                self.left_inner_line[1],
            ],
            10: [
                self.top_inner_line[0],
                self.middle_line[0],
                self.bottom_inner_line[0],
                self.middle_line[1],
            ],
            11: [
                self.middle_line[0],
                self.top_inner_line[1],
                self.middle_line[1],
                self.bottom_inner_line[1],
            ],
            12: [
                *self.bottom_inner_line,
                self.left_inner_line[1],
                self.right_inner_line[1],
            ],
        }
        self.line_width = 1
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2

    def build_court_reference(self) -> np.ndarray:
        """Create court reference image using the line positions."""
        court = np.zeros(
            (
                self.court_height + 2 * self.top_bottom_border,
                self.court_width + 2 * self.right_left_border,
            ),
            dtype=np.uint8,
        )
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        cv2.line(court, *self.net, 1, self.line_width)
        cv2.line(court, *self.top_inner_line, 1, self.line_width)
        cv2.line(court, *self.bottom_inner_line, 1, self.line_width)
        cv2.line(court, *self.left_court_line, 1, self.line_width)
        cv2.line(court, *self.right_court_line, 1, self.line_width)
        cv2.line(court, *self.left_inner_line, 1, self.line_width)
        cv2.line(court, *self.right_inner_line, 1, self.line_width)
        cv2.line(court, *self.middle_line, 1, self.line_width)
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))  # type: ignore[assignment]
        self.court = court
        return court

    def get_important_lines(self) -> list:  # type: ignore[type-arg]
        """Return all lines of the court as a flat list of point tuples."""
        lines = [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.net,
            *self.left_court_line,
            *self.right_court_line,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.middle_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
        ]
        return lines

    def get_extra_parts(self) -> list:  # type: ignore[type-arg]
        """Return extra reference parts (court center markers)."""
        parts = [self.top_extra_part, self.bottom_extra_part]
        return parts

    def get_court_mask(self, mask_type: int = 0) -> np.ndarray:
        """Get mask of the court region.

        Parameters
        ----------
        mask_type : int
            0 = full court, 1 = bottom half, 2 = top half, 3 = court without margins.
        """
        mask = np.ones_like(self.court)
        if mask_type == 1:  # Bottom half court
            mask[: self.net[0][1], :] = 0
        elif mask_type == 2:  # Top half court
            mask[self.net[0][1] :, :] = 0
        elif mask_type == 3:  # Court without margins
            mask[: self.baseline_top[0][1], :] = 0
            mask[self.baseline_bottom[0][1] :, :] = 0
            mask[:, : self.left_court_line[0][0]] = 0
            mask[:, self.right_court_line[0][0] :] = 0
        return mask
