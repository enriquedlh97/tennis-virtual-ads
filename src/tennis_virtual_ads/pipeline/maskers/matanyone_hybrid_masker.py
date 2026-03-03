"""Keyframe-hybrid occlusion masker — MatAnyone + optical flow propagation.

Runs the full MatAnyone matting model on keyframes (every N frames) and
warps the resulting alpha mask to intermediate frames using dense optical
flow (Farneback).  This amortises MatAnyone's ~1 fps cost across many
frames, yielding near-MatAnyone quality at a much higher effective
frame rate.

Dependencies
------------
Same as MatAnyone — requires ``matanyone`` and ``torch``.  The optical
flow computation uses only OpenCV (already a core dependency).
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)


class MatAnyoneHybridMasker(OcclusionMasker):
    """Keyframe MatAnyone + optical-flow interpolation.

    Parameters
    ----------
    device : str | None
        PyTorch device string.  Auto-selects CUDA if available.
    confidence_threshold : float
        Minimum detection score for person detections (passed to inner
        MatAnyone masker).
    keyframe_interval : int
        Run full MatAnyone inference every N frames.  Between keyframes,
        alpha masks are warped via dense optical flow.
    n_warmup : int
        MatAnyone warmup repetitions on first frame.
    use_yolo : bool
        Use YOLO for person detection in inner masker.
    yolo_model : str
        YOLO model filename.
    checkpoint_path : str | None
        Path to local MatAnyone checkpoint.
    max_internal_size : int
        MatAnyone internal resolution limit.
    box_padding : int
        Pixel padding on detection boxes.
    flow_scale : float
        Compute optical flow at this fraction of full resolution for
        speed.  ``0.5`` means half-resolution flow (default).
    """

    def __init__(
        self,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        keyframe_interval: int = 15,
        n_warmup: int = 5,
        use_yolo: bool = True,
        yolo_model: str = "yolo11n.pt",
        checkpoint_path: str | None = None,
        max_internal_size: int = -1,
        box_padding: int = 10,
        flow_scale: float = 0.5,
    ) -> None:
        from tennis_virtual_ads.pipeline.maskers.matanyone_masker import MatAnyoneMasker

        self._keyframe_interval = keyframe_interval
        self._flow_scale = max(0.1, min(1.0, flow_scale))

        logger.info(
            "MatAnyoneHybridMasker: keyframe_interval=%d, flow_scale=%.2f",
            self._keyframe_interval,
            self._flow_scale,
        )

        # Compose the inner MatAnyone masker (delegation, not inheritance).
        self._inner = MatAnyoneMasker(
            device=device,
            confidence_threshold=confidence_threshold,
            n_warmup=n_warmup,
            use_yolo=use_yolo,
            yolo_model=yolo_model,
            checkpoint_path=checkpoint_path,
            max_internal_size=max_internal_size,
            box_padding=box_padding,
        )

        # Flow / interpolation state
        self._frame_index: int = 0
        self._last_keyframe_index: int = -1
        self._prev_gray: np.ndarray | None = None
        self._current_alpha: np.ndarray | None = None  # latest alpha (keyframe or warped)

    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Generate an occlusion mask — keyframe or flow-warped.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image (OpenCV convention).

        Returns
        -------
        OcclusionMaskerResult
            ``mask`` is ``(H, W)`` float32 in ``[0, 1]``.
        """
        self._frame_index += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        is_keyframe = (
            self._current_alpha is None
            or (self._frame_index - self._last_keyframe_index) >= self._keyframe_interval
        )

        if is_keyframe:
            result = self._run_keyframe(frame, curr_gray)
        else:
            result = self._run_interpolation(curr_gray)

        self._prev_gray = curr_gray
        return result

    def reset_on_cut(self) -> None:
        """Reset on scene cut — clears flow state and inner MatAnyone."""
        self._inner.reset_on_cut()
        self._prev_gray = None
        self._current_alpha = None
        self._last_keyframe_index = -1
        logger.debug("MatAnyoneHybridMasker reset (scene cut).")

    # ------------------------------------------------------------------
    # Internal — keyframe (full MatAnyone)
    # ------------------------------------------------------------------

    def _run_keyframe(self, frame: np.ndarray, curr_gray: np.ndarray) -> OcclusionMaskerResult:
        """Run full MatAnyone inference and store as reference keyframe."""
        t0 = time.perf_counter()
        result = self._inner.mask(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._current_alpha = result["mask"].copy()
        self._last_keyframe_index = self._frame_index
        self._prev_gray = curr_gray

        logger.debug(
            "Hybrid keyframe #%d: %.0f ms, alpha_mean=%.4f",
            self._frame_index,
            elapsed_ms,
            float(self._current_alpha.mean()),
        )

        result["debug"]["hybrid_mode"] = "keyframe"
        result["debug"]["keyframe_ms"] = elapsed_ms
        return result

    # ------------------------------------------------------------------
    # Internal — interpolation (optical flow warp)
    # ------------------------------------------------------------------

    def _run_interpolation(self, curr_gray: np.ndarray) -> OcclusionMaskerResult:
        """Warp the last alpha mask using dense optical flow."""
        assert self._prev_gray is not None
        assert self._current_alpha is not None

        t0 = time.perf_counter()

        h, w = curr_gray.shape[:2]

        # Compute flow at reduced resolution for speed
        if self._flow_scale < 1.0:
            small_h = max(1, int(h * self._flow_scale))
            small_w = max(1, int(w * self._flow_scale))
            small_prev = cv2.resize(self._prev_gray, (small_w, small_h))
            small_curr = cv2.resize(curr_gray, (small_w, small_h))
        else:
            small_prev = self._prev_gray
            small_curr = curr_gray

        flow_out = np.zeros((*small_prev.shape, 2), dtype=np.float32)
        flow_small = cv2.calcOpticalFlowFarneback(
            small_prev,
            small_curr,
            flow_out,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Upscale flow to full resolution
        if self._flow_scale < 1.0:
            flow = cv2.resize(flow_small, (w, h)) / self._flow_scale
        else:
            flow = flow_small

        # Build remap coordinates: for each pixel (x, y), where did it come from?
        coords_x = (np.arange(w, dtype=np.float32)[np.newaxis, :] + flow[..., 0]).astype(np.float32)
        coords_y = (np.arange(h, dtype=np.float32)[:, np.newaxis] + flow[..., 1]).astype(np.float32)

        # Warp previous alpha mask
        warped_alpha = cv2.remap(
            self._current_alpha,
            coords_x,
            coords_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_alpha = np.clip(warped_alpha, 0.0, 1.0).astype(np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Update current alpha for next frame's flow warp
        self._current_alpha = warped_alpha

        logger.debug(
            "Hybrid interp #%d: %.1f ms, alpha_mean=%.4f",
            self._frame_index,
            elapsed_ms,
            float(warped_alpha.mean()),
        )

        return OcclusionMaskerResult(
            mask=warped_alpha,
            conf=0.9,  # slightly lower confidence for warped frames
            debug={
                "instance_count": -1,
                "scores": [],
                "hybrid_mode": "interpolation",
                "flow_ms": elapsed_ms,
                "alpha_mean": float(warped_alpha.mean()),
                "alpha_max": float(warped_alpha.max()),
            },
        )
