"""Video I/O wrappers around OpenCV VideoCapture and VideoWriter.

Design notes
------------
- Both classes are **context managers** so resources are always released.
- ``VideoReader`` is also an **iterator** that yields ``(frame_index, frame)``
  one frame at a time — it never loads the entire video into memory.
- ``VideoWriter`` uses the ``mp4v`` FourCC codec by default for broad .mp4
  compatibility on macOS / Linux / Windows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VideoReader
# ---------------------------------------------------------------------------


class VideoReader:
    """Lazy, frame-by-frame reader backed by ``cv2.VideoCapture``.

    Parameters
    ----------
    path : str
        Path to the input video file.
    start_frame : int
        First frame index to yield (0-based).
    max_frames : int | None
        Maximum number of frames to yield.  ``None`` → read until EOF.
    stride : int
        Yield every *N*-th frame.  ``1`` yields every frame.
    resize : tuple[int, int] | None
        ``(width, height)`` to resize each yielded frame.  ``None`` keeps
        the original resolution.
    """

    def __init__(
        self,
        path: str,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 1,
        resize: tuple[int, int] | None = None,
    ) -> None:
        self.path = path
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.stride = stride
        self.resize = resize
        self._capture: cv2.VideoCapture | None = None

    # -- read-only properties (available after open / __enter__) -----------

    @property
    def fps(self) -> float:
        """Frames per second reported by the video container."""
        self._assert_open()
        assert self._capture is not None  # narrowing for mypy (guarded by _assert_open)
        return self._capture.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        """Native frame width in pixels."""
        self._assert_open()
        assert self._capture is not None
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Native frame height in pixels."""
        self._assert_open()
        assert self._capture is not None
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        """Total frame count reported by the container (may be approximate)."""
        self._assert_open()
        assert self._capture is not None
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # -- context manager ---------------------------------------------------

    def open(self) -> VideoReader:
        """Open the underlying ``cv2.VideoCapture``."""
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            raise OSError(f"Cannot open video: {self.path}")
        logger.info(
            "Opened video: %s  |  %.1f fps  |  %dx%d  |  ~%d frames",
            self.path,
            self.fps,
            self.width,
            self.height,
            self.frame_count,
        )
        return self

    def close(self) -> None:
        """Release the underlying ``cv2.VideoCapture``."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoReader:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    # -- iterator ----------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` tuples one at a time."""
        self._assert_open()
        assert self._capture is not None  # narrowing for mypy

        # Seek to start_frame (cv2 seek is by frame position)
        if self.start_frame > 0:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        frames_yielded = 0
        current_index = self.start_frame

        while True:
            # Respect max_frames cap
            if self.max_frames is not None and frames_yielded >= self.max_frames:
                break

            ok, frame = self._capture.read()
            if not ok:
                break

            # Optional resize
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)

            yield current_index, frame
            frames_yielded += 1

            # Skip (stride - 1) frames via lightweight grab()
            for _ in range(self.stride - 1):
                if not self._capture.grab():
                    return
                current_index += 1

            current_index += 1

    # -- internals ---------------------------------------------------------

    def _assert_open(self) -> None:
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError(
                "VideoReader is not open. Use as a context manager or call .open() first."
            )


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------


class VideoWriter:
    """Frame-by-frame writer backed by ``cv2.VideoWriter``.

    Parameters
    ----------
    path : str
        Output file path (should end in ``.mp4``).
    fps : float
        Frames per second for the output video.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    codec : str
        FourCC codec string.  Default ``"mp4v"`` which works everywhere.
        The output can be re-encoded to H.264 via :func:`reencode_to_h264`.
    """

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None
        self._frames_written: int = 0

    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        return self._frames_written

    # -- context manager ---------------------------------------------------

    def open(self) -> VideoWriter:
        """Open the underlying ``cv2.VideoWriter``."""
        fourcc: int = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise OSError(f"Cannot open video writer: {self.path}")
        logger.info(
            "Opened writer: %s  |  %.1f fps  |  %dx%d  |  codec=%s",
            self.path,
            self.fps,
            self.width,
            self.height,
            self.codec,
        )
        return self

    def close(self) -> None:
        """Release the underlying ``cv2.VideoWriter``."""
        if self._writer is not None:
            self._writer.release()
            logger.info(
                "Closed writer: %s  |  %d frames written",
                self.path,
                self._frames_written,
            )
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    # -- write -------------------------------------------------------------

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame to the output video."""
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError(
                "VideoWriter is not open. Use as a context manager or call .open() first."
            )
        self._writer.write(frame)
        self._frames_written += 1


# ---------------------------------------------------------------------------
# Post-processing helper
# ---------------------------------------------------------------------------


def reencode_to_h264(video_path: str) -> bool:
    """Re-encode a video file to H.264 using ffmpeg (in-place).

    This is needed because OpenCV's ``mp4v`` codec produces files that
    many browsers and media players cannot play.  If ``ffmpeg`` is
    available on the system, this replaces the file with an H.264 version.

    Parameters
    ----------
    video_path : str
        Path to the video file to re-encode.

    Returns
    -------
    bool
        ``True`` if re-encoding succeeded, ``False`` if ffmpeg is not
        available or the re-encode failed (original file is preserved).
    """
    import os
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        logger.warning(
            "ffmpeg not found on PATH; output video may not play in "
            "all media players.  Install ffmpeg or re-encode manually."
        )
        return False

    temp_path = video_path + ".h264.tmp.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                "-an",  # no audio (our videos have none)
                temp_path,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(temp_path, video_path)
        logger.info("Re-encoded to H.264: %s", video_path)
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg re-encode failed: %s", exc.stderr.decode()[-200:])
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
