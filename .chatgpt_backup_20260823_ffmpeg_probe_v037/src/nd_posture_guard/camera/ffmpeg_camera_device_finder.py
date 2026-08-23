from __future__ import annotations

import os
import sys
from pathlib import Path


class FfmpegCameraDeviceFinder:
    """Resolve a Linux webcam path without opening it through OpenCV."""

    def __init__(self, maximum_index: int = 9) -> None:
        self._maximum_index = max(0, int(maximum_index))

    def find(self, configured_device: str) -> str:
        value = configured_device.strip()
        if value.lower() != "auto":
            return self._normalize_manual(value)

        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                "External ffmpeg camera capture is currently enabled only on Linux."
            )

        by_id = Path("/dev/v4l/by-id")
        if by_id.is_dir():
            candidates = sorted(by_id.glob("*-video-index0"))
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)

        for index in range(self._maximum_index + 1):
            candidate = Path(f"/dev/video{index}")
            if candidate.exists():
                return str(candidate)

        raise RuntimeError(
            "No Linux webcam device was found under /dev/v4l/by-id or /dev/video*."
        )

    @staticmethod
    def _normalize_manual(value: str) -> str:
        if value.isdigit():
            return f"/dev/video{int(value)}"
        if value.startswith("/dev/"):
            return value
        if sys.platform.startswith("linux"):
            candidate = Path(value)
            if candidate.exists():
                return os.path.realpath(candidate)
        return value
