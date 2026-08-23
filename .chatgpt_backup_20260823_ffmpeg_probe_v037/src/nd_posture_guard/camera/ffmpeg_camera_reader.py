from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
import time
from threading import Event, Lock, Thread

import numpy as np
from numpy.typing import NDArray


class FfmpegCameraReader:
    """Read V4L2 frames through an external ffmpeg process.

    No cv2.VideoCapture object exists in this class. A blocked V4L2 read can only
    block the child ffmpeg process; the monitoring worker keeps reading the latest
    completed frame from memory. If the pipe stops producing complete frames, the
    child process is killed and restarted.
    """

    START_TIMEOUT_SECONDS = 6.0
    FRAME_TIMEOUT_SECONDS = 2.0
    STALE_FRAME_SECONDS = 1.5
    RESTART_DELAY_SECONDS = 0.25

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        mirror: bool,
    ) -> None:
        self._device = str(device)
        self._width = int(width)
        self._height = int(height)
        self._fps = max(1, int(fps))
        self._mirror = bool(mirror)

        self._thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = Event()
        self._opened = Event()
        self._frame_ready = Event()
        self._lock = Lock()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_frame_time = 0.0
        self._open_error: str | None = None

    @property
    def device_label(self) -> str:
        return self._device

    def open(self) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("ffmpeg/V4L2 camera capture is available only on Linux.")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found in PATH.")

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._opened.clear()
        self._frame_ready.clear()
        self._open_error = None
        with self._lock:
            self._latest_frame = None
            self._latest_frame_time = 0.0

        self._thread = Thread(
            target=self._capture_loop,
            name="posture-ffmpeg-camera",
            daemon=True,
        )
        self._thread.start()

        if not self._opened.wait(self.START_TIMEOUT_SECONDS):
            self.close()
            raise RuntimeError(
                f"Timed out while starting ffmpeg camera capture for {self._device}."
            )
        if self._open_error is not None:
            error = self._open_error
            self.close()
            raise RuntimeError(error)
        if not self._frame_ready.wait(self.START_TIMEOUT_SECONDS):
            self.close()
            raise RuntimeError(
                f"ffmpeg opened {self._device}, but no complete camera frame arrived."
            )

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
        self._terminate_process()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._thread = None
        self._process = None

    def _capture_loop(self) -> None:
        first_attempt = True
        while not self._stop.is_set():
            process = self._start_process()
            if process is None:
                if first_attempt:
                    self._open_error = (
                        f"Could not start ffmpeg for webcam device {self._device}."
                    )
                    self._opened.set()
                    return
                self._stop.wait(self.RESTART_DELAY_SECONDS)
                continue

            self._process = process
            if first_attempt:
                first_attempt = False
                self._opened.set()

            try:
                self._read_process_frames(process)
            finally:
                self._terminate_specific_process(process)
                if self._process is process:
                    self._process = None

            if not self._stop.is_set():
                self._stop.wait(self.RESTART_DELAY_SECONDS)

        if first_attempt:
            self._opened.set()

    def _start_process(self) -> subprocess.Popen[bytes] | None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return None

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-f",
            "v4l2",
            "-framerate",
            str(self._fps),
            "-video_size",
            f"{self._width}x{self._height}",
            "-i",
            self._device,
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            )
        except OSError:
            return None

    def _read_process_frames(self, process: subprocess.Popen[bytes]) -> None:
        if process.stdout is None:
            return

        frame_size = self._width * self._height * 3
        file_descriptor = process.stdout.fileno()
        buffer = bytearray()

        while not self._stop.is_set():
            if process.poll() is not None:
                return

            readable, _, _ = select.select(
                [file_descriptor],
                [],
                [],
                self.FRAME_TIMEOUT_SECONDS,
            )
            if not readable:
                return

            try:
                chunk = os.read(file_descriptor, min(1024 * 1024, frame_size - len(buffer)))
            except OSError:
                return
            if not chunk:
                return
            buffer.extend(chunk)

            if len(buffer) < frame_size:
                continue

            raw = bytes(buffer[:frame_size])
            del buffer[:frame_size]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                self._height,
                self._width,
                3,
            )
            if self._mirror:
                frame = frame[:, ::-1, :].copy()
            else:
                frame = frame.copy()

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_time = time.monotonic()
            self._frame_ready.set()

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        self._terminate_specific_process(process)
        if self._process is process:
            self._process = None

    @staticmethod
    def _terminate_specific_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.6)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=0.6)
            except (OSError, subprocess.TimeoutExpired):
                pass
