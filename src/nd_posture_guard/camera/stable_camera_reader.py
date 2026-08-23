from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np
from numpy.typing import NDArray


class StableCameraReader:
    """Capture frames on a dedicated thread and expose only the latest frame.

    Monitoring never calls VideoCapture.read() directly.  If a V4L2 read stalls,
    the Qt/monitoring worker remains responsive instead of being held inside the
    native OpenCV call.
    """

    START_TIMEOUT_SECONDS = 4.0
    STALE_FRAME_SECONDS = 1.5

    def __init__(
        self,
        device: str | int,
        width: int,
        height: int,
        fps: int,
        mirror: bool,
    ) -> None:
        self._device = self._normalize_device(device)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._mirror = bool(mirror)
        self._capture: cv2.VideoCapture | None = None
        self._thread: Thread | None = None
        self._stop = Event()
        self._opened = Event()
        self._frame_ready = Event()
        self._lock = Lock()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_frame_time = 0.0
        self._open_error: str | None = None

    @property
    def device_label(self) -> str:
        return str(self._device)

    def open(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._opened.clear()
        self._frame_ready.clear()
        self._open_error = None
        with self._lock:
            self._latest_frame = None
            self._latest_frame_time = 0.0

        self._thread = Thread(target=self._capture_loop, name="posture-camera", daemon=True)
        self._thread.start()
        if not self._opened.wait(self.START_TIMEOUT_SECONDS):
            self.close()
            raise RuntimeError(f"Timed out while opening webcam device {self._device}.")
        if self._open_error is not None:
            error = self._open_error
            self.close()
            raise RuntimeError(error)
        self._frame_ready.wait(1.5)

    def read(self) -> NDArray[np.uint8] | None:
        with self._lock:
            frame = self._latest_frame
            frame_time = self._latest_frame_time
            if frame is None:
                return None
            if time.monotonic() - frame_time > self.STALE_FRAME_SECONDS:
                return None
            return frame.copy()

    def close(self) -> None:
        self._stop.set()
        capture = self._capture
        if capture is not None:
            try:
                capture.release()
            except cv2.error:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._capture = None

    def _capture_loop(self) -> None:
        first_open = True
        while not self._stop.is_set():
            capture = self._open_capture()
            self._capture = capture
            if not capture.isOpened():
                capture.release()
                self._capture = None
                if first_open:
                    self._open_error = (
                        f"Cannot open webcam device {self._device}. "
                        "Check camera connection and permissions."
                    )
                    self._opened.set()
                    return
                self._stop.wait(0.35)
                continue

            self._configure(capture)
            if first_open:
                first_open = False
                self._opened.set()

            failures = 0
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= 12:
                        break
                    self._stop.wait(0.02)
                    continue
                failures = 0
                if self._mirror:
                    frame = cv2.flip(frame, 1)
                with self._lock:
                    self._latest_frame = frame
                    self._latest_frame_time = time.monotonic()
                self._frame_ready.set()

            try:
                capture.release()
            except cv2.error:
                pass
            self._capture = None
            if not self._stop.is_set():
                self._stop.wait(0.20)

        if first_open:
            self._opened.set()

    def _open_capture(self) -> cv2.VideoCapture:
        if sys.platform.startswith("linux"):
            index = self._linux_index(self._device)
            if index is not None:
                return cv2.VideoCapture(index, cv2.CAP_V4L2)
        return cv2.VideoCapture(self._device)

    def _configure(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    @classmethod
    def _normalize_device(cls, device: str | int) -> str | int:
        if isinstance(device, int):
            return device
        if sys.platform.startswith("linux"):
            index = cls._linux_index(device)
            if index is not None:
                return index
        return device

    @staticmethod
    def _linux_index(device: str | int) -> int | None:
        if isinstance(device, int):
            return device
        if not isinstance(device, str) or not device.startswith("/dev/"):
            return None
        try:
            name = Path(os.path.realpath(device)).name
        except OSError:
            name = Path(device).name
        if not name.startswith("video"):
            return None
        suffix = name.removeprefix("video")
        return int(suffix) if suffix.isdigit() else None
