"""Ad warping and naive alpha compositing.

The ``AdPlacer`` takes an RGBA ad image, a court-to-image homography,
and the 4 ad corners in court-reference space, then:

1. Projects the court-ref corners through H to get image-space corners.
2. Computes a perspective transform from ad-pixel space to image space.
3. Warps the ad into a full-frame-sized image.
4. Alpha-composites the warped ad onto the video frame.

No occlusion handling yet — the ad is drawn *over* everything (players,
ball, etc.).  Occlusion masking will be layered on top in a later step.
"""

from __future__ import annotations

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.placer.placement import (
    PlacementSpec,
    PreparedPlacement,
    get_precomputed_court_corners,
)


class AdPlacer:
    """Warp an RGBA ad image onto the court surface via homography.

    The class is stateless — it can be reused across frames.
    """

    def warp(
        self,
        ad_rgba: np.ndarray,
        H: np.ndarray,
        placement_spec: PreparedPlacement | PlacementSpec,
        frame_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp the ad into image space using the homography.

        Parameters
        ----------
        ad_rgba : np.ndarray
            The ad image as ``(H_ad, W_ad, 4)`` uint8 BGRA array.
        H : np.ndarray
            3x3 court-to-image homography matrix.
        placement_spec : PreparedPlacement | PlacementSpec
            A *prepared* placement object with precomputed court-reference corners.
            Passing a raw ``PlacementSpec`` raises to avoid hidden per-frame geometry.
        frame_shape : tuple[int, int, int]
            Shape of the output video frame (H, W, C).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(warped_rgba, warped_mask)`` where:
            - ``warped_rgba`` is ``(frame_H, frame_W, 4)`` uint8 BGRA image
              of the warped ad (transparent where no ad).
            - ``warped_mask`` is ``(frame_H, frame_W)`` float32 in [0, 1]
              from the ad's alpha channel.
        """
        frame_height, frame_width = frame_shape[:2]
        ad_height, ad_width = ad_rgba.shape[:2]

        court_corners = get_precomputed_court_corners(placement_spec)

        # --- 1. Project court-ref corners to image space ------------------
        court_points = court_corners.reshape(-1, 1, 2).astype(np.float32)
        image_points = cv2.perspectiveTransform(court_points, H)
        image_points = image_points.reshape(-1, 2)

        # --- 2. Ad source corners (pixel space of the ad image) -----------
        # Use (width - 1, height - 1) so corners map to the actual image extents.
        ad_source_corners = np.array(
            [
                [0, 0],
                [ad_width - 1, 0],
                [ad_width - 1, ad_height - 1],
                [0, ad_height - 1],
            ],
            dtype=np.float32,
        )

        # --- 3. Perspective transform: ad pixels -> image pixels ----------
        perspective_matrix = cv2.getPerspectiveTransform(ad_source_corners, image_points)

        # --- 4. Warp the full RGBA ad into frame-sized canvas -------------
        warped_rgba = cv2.warpPerspective(
            ad_rgba,
            perspective_matrix,
            (frame_width, frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # --- 5. Split into RGB + normalised alpha mask --------------------
        warped_mask = warped_rgba[:, :, 3].astype(np.float32) / 255.0

        return warped_rgba, warped_mask

    @staticmethod
    def composite(
        frame: np.ndarray,
        warped_rgba: np.ndarray,
        warped_mask: np.ndarray,
    ) -> None:
        """Alpha-composite the warped ad onto *frame* (mutates in-place).

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR video frame.
        warped_rgba : np.ndarray
            ``(H, W, 4)`` uint8 BGRA warped ad image.
        warped_mask : np.ndarray
            ``(H, W)`` float32 alpha mask in [0, 1].
        """
        warped_bgr = warped_rgba[:, :, :3]

        # Expand mask to 3 channels for broadcasting.
        alpha_3channel = warped_mask[:, :, np.newaxis]

        # Blend: out = frame * (1 - alpha) + ad * alpha
        blended = (
            frame.astype(np.float32) * (1.0 - alpha_3channel)
            + warped_bgr.astype(np.float32) * alpha_3channel
        )
        np.copyto(frame, blended.astype(np.uint8))
