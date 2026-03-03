"""SAM2 Video Predictor masker — offline, highest-quality SAM2 masks.

Uses SAM2's **video predictor** API (``build_sam2_video_predictor`` +
``propagate_in_video()``) for temporally coherent segmentation with a
built-in memory bank.  All masks are pre-computed before the pipeline
starts rendering, so per-frame ``mask()`` calls are O(1) dict lookups.

This is an **offline-only** masker (requires the full video upfront)
intended for:
    - Demo / comparison renders (highest mask quality)
    - Generating training labels for future model distillation

Players are auto-detected on a configurable prompt frame using YOLO
(preferred) or Mask R-CNN (fallback), then converted to SAM2 point
prompts (box-center clicks).

Dependencies
------------
Requires ``sam2`` and optionally ``ultralytics`` (for YOLO prompting).
Install via::

    uv pip install -e ".[sam2]"
    uv pip install -e ".[yolo]"  # optional, for YOLO prompting
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)


def _check_sam2() -> None:
    """Raise a clear error if sam2 is not installed."""
    try:
        import sam2  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "SAM2VideoMasker requires sam2.  "
            "Install the sam2 extras:\n\n"
            '    uv pip install -e ".[sam2]"\n'
        ) from error


# Map friendly CLI names to HuggingFace model IDs.
# Used by ``build_sam2_video_predictor_hf`` for auto-download.
_SAM2_HF_MODEL_IDS: dict[str, str] = {
    "sam2.1_hiera_t": "facebook/sam2.1-hiera-tiny",
    "sam2.1_hiera_s": "facebook/sam2.1-hiera-small",
    "sam2.1_hiera_b+": "facebook/sam2.1-hiera-base-plus",
    "sam2.1_hiera_l": "facebook/sam2.1-hiera-large",
}

# Map friendly CLI names to sam2 config YAML paths (used only with explicit checkpoint_path).
_SAM2_MODEL_CONFIGS: dict[str, str] = {
    "sam2.1_hiera_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


class SAM2VideoMasker(OcclusionMasker):
    """Offline SAM2 video predictor masker with YOLO/Mask R-CNN auto-prompting.

    All masks are pre-computed during ``__init__``.  Subsequent ``mask()``
    calls are O(1) dict lookups.

    Parameters
    ----------
    video_path : str
        Path to the input video file.  Required upfront for frame extraction.
    device : str | None
        PyTorch device string.  Auto-selects CUDA if available.
    model_cfg : str
        SAM2 model configuration name.  Default ``"sam2.1_hiera_l"`` (large,
        310M params) for best quality.
    checkpoint_path : str | None
        Path to a local SAM2 checkpoint file.  If ``None``, auto-downloads
        from ``facebook/sam2.1-hiera-large`` on HuggingFace Hub.
    confidence_threshold : float
        Minimum YOLO/MRCNN detection score to include a person.
    use_yolo : bool
        Use YOLO for person detection.  Falls back to Mask R-CNN if YOLO
        is unavailable or ``use_yolo=False``.
    yolo_model : str
        YOLO model filename (auto-downloaded by ultralytics).
    prompt_frame_idx : int
        Frame index on which to run person detection for auto-prompting.
    """

    def __init__(
        self,
        video_path: str,
        device: str | None = None,
        model_cfg: str = "sam2.1_hiera_l",
        checkpoint_path: str | None = None,
        confidence_threshold: float = 0.5,
        use_yolo: bool = True,
        yolo_model: str = "yolo11n.pt",
        prompt_frame_idx: int = 0,
    ) -> None:
        _check_sam2()

        import torch

        # Suppress noisy SAM2 root-logger messages.
        _sam2_suppressed = frozenset(
            {
                "Computing image embeddings for the provided image...",
                "Image embeddings computed.",
                "For numpy array image, we assume (HxWxC) format",
                "Loaded checkpoint sucessfully",
            }
        )

        class _SAM2LogFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return record.getMessage() not in _sam2_suppressed

        logging.getLogger().addFilter(_SAM2LogFilter())

        self._confidence_threshold = confidence_threshold
        self._prompt_frame_idx = prompt_frame_idx

        # Auto-select device.
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Validate model_cfg name.
        if model_cfg not in _SAM2_HF_MODEL_IDS:
            raise ValueError(
                f"Unknown SAM2 model config '{model_cfg}'. "
                f"Available: {list(_SAM2_HF_MODEL_IDS.keys())}"
            )
        self._model_cfg = model_cfg
        self._checkpoint_path = checkpoint_path

        # --- Load person detector -------------------------------------------
        from tennis_virtual_ads.pipeline.maskers._person_detector import (
            load_mrcnn,
            load_yolo,
            yolo_available,
        )

        self._use_yolo = use_yolo and yolo_available()
        self._yolo_model_obj: Any = None
        self._mrcnn_model: Any = None
        self._mrcnn_transforms: Any = None

        if self._use_yolo:
            self._yolo_model_obj = load_yolo(yolo_model)
        else:
            self._mrcnn_model, self._mrcnn_transforms = load_mrcnn(self._device)

        # --- Extract frames to temp dir -------------------------------------
        self._temp_dir: str | None = None
        frame_dir, total_frames, prompt_frame_bgr = self._extract_frames(video_path)

        # --- Auto-detect persons on prompt frame ----------------------------
        boxes, scores = self._detect_persons(prompt_frame_bgr)
        logger.info(
            "SAM2VideoMasker: auto-detected %d persons on frame %d (scores=%s)",
            len(boxes),
            self._prompt_frame_idx,
            [f"{s:.2f}" for s in scores],
        )

        # --- Build SAM2 video predictor ------------------------------------
        load_start = time.perf_counter()

        if self._checkpoint_path is not None:
            # Explicit local checkpoint — use config + ckpt_path.
            config_file = _SAM2_MODEL_CONFIGS[self._model_cfg]
            logger.info(
                "Loading SAM2 video predictor '%s' (config=%s, ckpt=%s) on %s ...",
                model_cfg,
                config_file,
                self._checkpoint_path,
                self._device,
            )
            from sam2.build_sam import build_sam2_video_predictor

            self._predictor = build_sam2_video_predictor(
                config_file,
                self._checkpoint_path,
                device=str(self._device),
            )
        else:
            # Auto-download from HuggingFace Hub.
            hf_model_id = _SAM2_HF_MODEL_IDS[self._model_cfg]
            logger.info(
                "Loading SAM2 video predictor '%s' from HuggingFace (%s) on %s ...",
                model_cfg,
                hf_model_id,
                self._device,
            )
            from sam2.build_sam import build_sam2_video_predictor_hf

            self._predictor = build_sam2_video_predictor_hf(
                hf_model_id,
                device=str(self._device),
            )

        load_elapsed = time.perf_counter() - load_start
        logger.info("SAM2 video predictor loaded in %.1f s", load_elapsed)

        # --- Initialize state and propagate --------------------------------
        self._masks: dict[int, np.ndarray] = {}
        self._frame_count: int = 0

        if len(boxes) == 0:
            logger.warning(
                "SAM2VideoMasker: no persons detected on frame %d. All masks will be empty.",
                self._prompt_frame_idx,
            )
            # Store empty masks for all frames.
            h, w = prompt_frame_bgr.shape[:2]
            for i in range(total_frames):
                self._masks[i] = np.zeros((h, w), dtype=np.float32)
        else:
            self._propagate_all(frame_dir, total_frames, boxes, prompt_frame_bgr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Return the pre-computed mask for the current frame.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image (unused — masks are pre-computed).

        Returns
        -------
        OcclusionMaskerResult
        """
        idx = self._frame_count
        self._frame_count += 1

        if idx in self._masks:
            m = self._masks[idx]
            return OcclusionMaskerResult(
                mask=m,
                conf=1.0,
                debug={"frame_idx": idx, "mode": "precomputed"},
            )

        # Beyond pre-computed range — return empty mask.
        h, w = frame.shape[:2]
        return OcclusionMaskerResult(
            mask=np.zeros((h, w), dtype=np.float32),
            conf=0.0,
            debug={"frame_idx": idx, "mode": "out_of_range"},
        )

    def reset_on_cut(self) -> None:
        """No-op for offline masker (masks are already pre-computed)."""

    # ------------------------------------------------------------------
    # Internal — frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(self, video_path: str) -> tuple[str, int, np.ndarray]:
        """Extract all video frames as JPEG files into a temp directory.

        Returns
        -------
        tuple[str, int, np.ndarray]
            ``(frame_dir, total_frames, prompt_frame_bgr)`` where
            *frame_dir* is the path to the temp directory containing
            JPEG frames, *total_frames* is the count, and
            *prompt_frame_bgr* is the BGR image at ``prompt_frame_idx``.
        """
        logger.info("SAM2VideoMasker: extracting frames from '%s' ...", video_path)
        extract_start = time.perf_counter()

        self._temp_dir = tempfile.mkdtemp(prefix="sam2_video_frames_")
        frame_dir = self._temp_dir

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        prompt_frame_bgr: np.ndarray | None = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # SAM2 video predictor expects JPEG frames in a directory.
            out_path = str(Path(frame_dir) / f"{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            if frame_idx == self._prompt_frame_idx:
                prompt_frame_bgr = frame.copy()
            frame_idx += 1

        cap.release()
        total_frames = frame_idx

        if prompt_frame_bgr is None:
            raise ValueError(
                f"prompt_frame_idx={self._prompt_frame_idx} is out of range "
                f"(video has {total_frames} frames)"
            )

        extract_elapsed = time.perf_counter() - extract_start
        logger.info(
            "SAM2VideoMasker: extracted %d frames in %.1f s to %s",
            total_frames,
            extract_elapsed,
            frame_dir,
        )
        return frame_dir, total_frames, prompt_frame_bgr

    # ------------------------------------------------------------------
    # Internal — propagation
    # ------------------------------------------------------------------

    def _propagate_all(
        self,
        frame_dir: str,
        total_frames: int,
        boxes: list[list[float]],
        prompt_frame_bgr: np.ndarray,
    ) -> None:
        """Run SAM2 video predictor on all frames.

        1. Initialize inference state from extracted frames.
        2. Add point prompts (box centers) for each detected person.
        3. Propagate masks across entire video.
        4. Store combined binary masks in ``self._masks``.
        """
        import torch

        frame_h, frame_w = prompt_frame_bgr.shape[:2]

        # Initialize inference state.
        logger.info("SAM2VideoMasker: initializing inference state ...")
        inference_state = self._predictor.init_state(video_path=frame_dir)

        # Add point prompts for each detected person.
        for obj_id, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            points = np.array([[cx, cy]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

            with torch.inference_mode():
                if str(self._device).startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        self._predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=self._prompt_frame_idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )
                else:
                    self._predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=self._prompt_frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                    )

        # Propagate masks across all frames.
        logger.info(
            "SAM2VideoMasker: propagating masks across %d frames ...",
            total_frames,
        )
        prop_start = time.perf_counter()

        with torch.inference_mode():
            if str(self._device).startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    video_segments = {}
                    for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(
                        inference_state
                    ):
                        video_segments[frame_idx] = {
                            oid: (mask_logits[i] > 0.0).cpu().numpy().astype(np.float32)
                            for i, oid in enumerate(obj_ids)
                        }
            else:
                video_segments = {}
                for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(
                    inference_state
                ):
                    video_segments[frame_idx] = {
                        oid: (mask_logits[i] > 0.0).cpu().numpy().astype(np.float32)
                        for i, oid in enumerate(obj_ids)
                    }

        prop_elapsed = time.perf_counter() - prop_start
        logger.info(
            "SAM2VideoMasker: propagation complete in %.1f s (%.1f fps)",
            prop_elapsed,
            total_frames / prop_elapsed if prop_elapsed > 0 else 0,
        )

        # Combine per-object masks into a single union mask per frame.
        for fidx in range(total_frames):
            if fidx in video_segments:
                obj_masks = list(video_segments[fidx].values())
                if obj_masks:
                    # Each mask is (1, H, W) — squeeze to (H, W) then union.
                    squeezed = [np.squeeze(m) for m in obj_masks]
                    union = np.maximum.reduce(squeezed)
                    self._masks[fidx] = union.astype(np.float32)
                else:
                    self._masks[fidx] = np.zeros((frame_h, frame_w), dtype=np.float32)
            else:
                self._masks[fidx] = np.zeros((frame_h, frame_w), dtype=np.float32)

        logger.info(
            "SAM2VideoMasker: stored masks for %d frames (non-empty: %d)",
            total_frames,
            sum(1 for m in self._masks.values() if m.max() > 0),
        )

        # Reset predictor state to free VRAM.
        self._predictor.reset_state(inference_state)

    # ------------------------------------------------------------------
    # Internal — person detection
    # ------------------------------------------------------------------

    def _detect_persons(self, frame: np.ndarray) -> tuple[list[list[float]], list[float]]:
        """Detect persons using YOLO or Mask R-CNN."""
        from tennis_virtual_ads.pipeline.maskers._person_detector import (
            detect_persons_mrcnn,
            detect_persons_yolo,
        )

        if self._use_yolo and self._yolo_model_obj is not None:
            return detect_persons_yolo(frame, self._yolo_model_obj, self._confidence_threshold)
        elif self._mrcnn_model is not None:
            return detect_persons_mrcnn(
                frame,
                self._mrcnn_model,
                self._mrcnn_transforms,
                self._device,
                self._confidence_threshold,
            )
        return [], []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Clean up extracted frames temp directory."""
        if self._temp_dir is not None and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug("SAM2VideoMasker: cleaned up temp dir %s", self._temp_dir)
            except OSError:
                pass
