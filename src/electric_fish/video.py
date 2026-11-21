"""Random-access frame reading for the tracking videos."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Thin wrapper over ``cv2.VideoCapture`` with indexing and iteration.

    Frames are returned as RGB ``uint8`` arrays (or 2-D grayscale with ``gray=True``).
    Indices are 0-based video frame indices. CSV frame ``k`` is video index ``k`` (checked on
    all four sessions with :func:`electric_fish.tracking.frame_alignment`).

    >>> with VideoReader("data/video2024-02-20T12_56_22.avi") as v:
    ...     img = v[100]
    """

    def __init__(self, path: str | Path, gray: bool = False):
        self.path = Path(path)
        self.gray = gray
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"Could not open video {self.path}")
        self._next = 0  # index of the frame the decoder will return next

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

    def __len__(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _convert(self, frame: np.ndarray) -> np.ndarray:
        code = cv2.COLOR_BGR2GRAY if self.gray else cv2.COLOR_BGR2RGB
        return cv2.cvtColor(frame, code)

    def __getitem__(self, index: int) -> np.ndarray:
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"frame {index} out of range (0..{n - 1})")
        if index != self._next:  # avoid an expensive seek for sequential access
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise OSError(f"Failed to decode frame {index} of {self.path}")
        self._next = index + 1
        return self._convert(frame)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_range(0, len(self))

    def iter_range(self, start: int, stop: int, step: int = 1) -> Iterator[np.ndarray]:
        for i in range(start, stop, step):
            yield self[i]

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
