from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraReader:
    def __init__(
        self,
        device: str | int,
        width: int,
        height: int,
        fps: int,
        mirror: bool,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._mirror = mirror
        self._capture: cv2.VideoCapture | None = None

    @property
    def device_label(self) -> str:
        return str(self._device)

    def open(self) -> None:
        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Cannot open webcam device {self._device}. "
                "Check camera connection and permissions."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture

    def read(self) -> NDArray[np.uint8] | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if self._mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _open_capture(self) -> cv2.VideoCapture:
        if isinstance(self._device, str) and self._device.startswith("/dev/"):
            capture = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(self._device)
