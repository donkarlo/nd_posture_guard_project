from __future__ import annotations

from threading import Lock

import numpy as np
from numpy.typing import NDArray


class CachedUsableCameraReader:
    """Return only usable frames and retain the latest good frame across transient black frames."""

    def __init__(self, reader) -> None:
        self._reader = reader
        self._lock = Lock()
        self._last_usable_frame: NDArray[np.uint8] | None = None

    @property
    def device_label(self) -> str:
        return self._reader.device_label

    def open(self) -> None:
        self._reader.open()
        initial = self._reader.read()
        if self._frame_is_usable(initial):
            with self._lock:
                self._last_usable_frame = initial.copy()

    def close(self) -> None:
        self._reader.close()

    def read(self) -> NDArray[np.uint8] | None:
        frame = self._reader.read()
        if self._frame_is_usable(frame):
            with self._lock:
                self._last_usable_frame = frame.copy()
            return frame

        with self._lock:
            cached = self._last_usable_frame
            return None if cached is None else cached.copy()

    @staticmethod
    def _frame_is_usable(frame) -> bool:
        if frame is None or getattr(frame, "size", 0) <= 0:
            return False

        sampled = frame[::16, ::16]
        if sampled.size == 0:
            return False
        maximum = int(sampled.max())
        variation = float(sampled.std())
        return maximum >= 8 or variation >= 2.0
