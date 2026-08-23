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


class ProbedFfmpegCameraReader:
    """Continuously read one already-probed V4L2 mode through ffmpeg."""

    START_TIMEOUT_SECONDS = 6.0
    FRAME_TIMEOUT_SECONDS = 2.0
    STALE_FRAME_SECONDS = 1.5
    RESTART_DELAY_SECONDS = 0.25

    def __init__(
        self,
        device: str,
        input_format: str | None,
        width: int,
        height: int,
        fps: int,
        mirror: bool,
    ) -> None:
        self._device = str(device)
        self._input_format = input_format
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
        self._last_ffmpeg_error = ""

    @property
    def device_label(self) -> str:
        mode = self._input_format or "auto"
        return f"{self._device} ({mode})"

    def open(self) -> None:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("ffmpeg/V4L2 camera capture is available only on Linux.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg was not found in PATH.")
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._opened.clear()
        self._frame_ready.clear()
        self._open_error = None
        self._last_ffmpeg_error = ""
        with self._lock:
            self._latest_frame = None
            self._latest_frame_time = 0.0

        self._thread = Thread(
            target=self._capture_loop,
            name="posture-probed-ffmpeg-camera",
            daemon=True,
        )
        self._thread.start()

        if not self._opened.wait(self.START_TIMEOUT_SECONDS):
            self.close()
            raise RuntimeError(f"Timed out while starting camera {self.device_label}.")
        if self._open_error is not None:
            error = self._open_error
            self.close()
            raise RuntimeError(error)
        if not self._frame_ready.wait(self.START_TIMEOUT_SECONDS):
            detail = self._last_ffmpeg_error or "no complete frame arrived"
            self.close()
            raise RuntimeError(f"Camera {self.device_label} produced no frame: {detail}")

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
                    self._open_error = f"Could not start ffmpeg for {self.device_label}."
                    self._opened.set()
                    return
                self._stop.wait(self.RESTART_DELAY_SECONDS)
                continue

            self._process = process
            if first_attempt:
                first_attempt = False
                self._opened.set()

            got_frame = False
            try:
                got_frame = self._read_process_frames(process)
            finally:
                self._terminate_specific_process(process)
                self._capture_stderr(process)
                if self._process is process:
                    self._process = None

            if not got_frame and not self._stop.is_set():
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
        ]
        if self._input_format is not None:
            command.extend(("-input_format", self._input_format))
        command.extend(
            (
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
            )
        )

        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            self._last_ffmpeg_error = str(exc)
            return None

    def _read_process_frames(self, process: subprocess.Popen[bytes]) -> bool:
        if process.stdout is None:
            return False

        frame_size = self._width * self._height * 3
        file_descriptor = process.stdout.fileno()
        buffer = bytearray()
        got_frame = False

        while not self._stop.is_set():
            if process.poll() is not None:
                return got_frame

            readable, _, _ = select.select(
                [file_descriptor],
                [],
                [],
                self.FRAME_TIMEOUT_SECONDS,
            )
            if not readable:
                self._last_ffmpeg_error = "camera pipe timed out"
                return got_frame

            try:
                chunk = os.read(
                    file_descriptor,
                    min(1024 * 1024, frame_size - len(buffer)),
                )
            except OSError as exc:
                self._last_ffmpeg_error = str(exc)
                return got_frame
            if not chunk:
                return got_frame
            buffer.extend(chunk)

            if len(buffer) < frame_size:
                continue

            raw = bytes(buffer[:frame_size])
            buffer.clear()
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
            got_frame = True

        return got_frame

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        self._terminate_specific_process(process)
        self._capture_stderr(process)
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

    def _capture_stderr(self, process: subprocess.Popen[bytes]) -> None:
        if process.stderr is None:
            return
        try:
            data = process.stderr.read()
        except OSError:
            return
        if not data:
            return
        text = data.decode("utf-8", errors="replace").strip()
        if text:
            self._last_ffmpeg_error = " ".join(text.split())[-700:]
