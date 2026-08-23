from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


class ProbedFfmpegCameraDeviceFinder:
    """Choose a Linux V4L2 node only after ffmpeg proves it returns a frame."""

    PROBE_TIMEOUT_SECONDS = 3.0
    INPUT_FORMATS: tuple[str | None, ...] = (None, "mjpeg", "yuyv422")

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        maximum_index: int = 9,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._fps = max(1, int(fps))
        self._maximum_index = max(0, int(maximum_index))

    def find(self, configured_device: str) -> tuple[str, str | None]:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("ffmpeg/V4L2 camera probing is available only on Linux.")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found in PATH.")

        configured = configured_device.strip()
        if configured.lower() != "auto":
            candidates = [self._normalize_manual(configured)]
        else:
            candidates = self._candidate_devices()

        if not candidates:
            raise RuntimeError("No /dev/video* camera device was found.")

        errors: list[str] = []
        for device in candidates:
            for input_format in self.INPUT_FORMATS:
                ok, error = self._probe(ffmpeg, device, input_format)
                if ok:
                    return device, input_format
                if error:
                    format_name = input_format or "auto"
                    errors.append(f"{device} [{format_name}]: {error}")

        detail = " | ".join(errors[-6:])
        if detail:
            raise RuntimeError(f"No webcam produced a usable ffmpeg frame. {detail}")
        raise RuntimeError("No webcam produced a usable ffmpeg frame.")

    def _candidate_devices(self) -> list[str]:
        candidates: list[str] = []
        seen_real_paths: set[str] = set()

        by_id = Path("/dev/v4l/by-id")
        if by_id.is_dir():
            for path in sorted(by_id.glob("*-video-index0")):
                if path.exists():
                    self._append_candidate(path, candidates, seen_real_paths)

        for index in range(self._maximum_index + 1):
            path = Path(f"/dev/video{index}")
            if path.exists():
                self._append_candidate(path, candidates, seen_real_paths)

        return candidates

    @staticmethod
    def _append_candidate(
        path: Path,
        candidates: list[str],
        seen_real_paths: set[str],
    ) -> None:
        try:
            real_path = os.path.realpath(path)
        except OSError:
            real_path = str(path)
        if real_path in seen_real_paths:
            return
        seen_real_paths.add(real_path)
        candidates.append(str(path))

    def _probe(
        self,
        ffmpeg: str,
        device: str,
        input_format: str | None,
    ) -> tuple[bool, str]:
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
        if input_format is not None:
            command.extend(("-input_format", input_format))
        command.extend(
            (
                "-framerate",
                str(self._fps),
                "-video_size",
                f"{self._width}x{self._height}",
                "-i",
                device,
                "-frames:v",
                "1",
                "-an",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            )
        )

        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "probe timed out"
        except OSError as exc:
            return False, str(exc)

        expected = self._width * self._height * 3
        if result.returncode == 0 and len(result.stdout) >= expected:
            return True, ""

        error = result.stderr.decode("utf-8", errors="replace").strip()
        if not error:
            error = f"ffmpeg exited {result.returncode}, bytes={len(result.stdout)}/{expected}"
        return False, " ".join(error.split())[-500:]

    @staticmethod
    def _normalize_manual(value: str) -> str:
        if value.isdigit():
            return f"/dev/video{int(value)}"
        return value
