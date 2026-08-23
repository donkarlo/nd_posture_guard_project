from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class StableCameraDeviceFinder:
    """Find a webcam while using numeric V4L2 indexes on Linux.

    Some OpenCV builds cannot open V4L2 devices by a '/dev/videoN' string when a
    backend is explicitly requested.  Numeric indexes avoid the name-based
    fallback path that produced the recurring VIDEOIO/V4L2 warnings and timing
    dependent capture behavior on this machine.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        maximum_index: int = 9,
        test_frame_count: int = 4,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._maximum_index = int(maximum_index)
        self._test_frame_count = max(2, int(test_frame_count))

    def find(self, configured_device: str) -> str | int:
        normalized = configured_device.strip()
        if normalized.lower() != "auto":
            return self._parse_manual_device(normalized)

        best_device: str | int | None = None
        best_score = float("-inf")
        for candidate in self._candidate_devices():
            score = self._score_device(candidate)
            if score is not None and score > best_score:
                best_device = candidate
                best_score = score

        if best_device is None:
            raise RuntimeError(
                "No usable webcam was found automatically. Check camera connection and permissions."
            )
        return best_device

    def _candidate_devices(self) -> list[str | int]:
        if sys.platform.startswith("linux"):
            indexes = self._linux_video_indexes()
            if indexes:
                return indexes
        return list(range(self._maximum_index + 1))

    def _linux_video_indexes(self) -> list[int]:
        indexes: list[int] = []
        seen: set[int] = set()

        candidates: list[Path] = []
        by_id = Path("/dev/v4l/by-id")
        if by_id.is_dir():
            candidates.extend(sorted(by_id.glob("*-video-index0")))
        candidates.extend(sorted(Path("/dev").glob("video*"), key=self._video_path_sort_key))

        for path in candidates:
            index = self._linux_index_from_path(path)
            if index is None or index in seen or index > self._maximum_index:
                continue
            seen.add(index)
            indexes.append(index)
        return indexes

    def _score_device(self, device: str | int) -> float | None:
        capture = self._open_capture(device)
        if not capture.isOpened():
            capture.release()
            return None
        try:
            self._configure(capture)
            frames: list[NDArray[np.uint8]] = []
            for _ in range(self._test_frame_count):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size > 0:
                    frames.append(frame)
            if not frames:
                return None
            return self._frame_sequence_score(frames)
        finally:
            capture.release()

    def _open_capture(self, device: str | int) -> cv2.VideoCapture:
        if sys.platform.startswith("linux") and isinstance(device, int):
            return cv2.VideoCapture(device, cv2.CAP_V4L2)
        return cv2.VideoCapture(device)

    def _configure(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _frame_sequence_score(self, frames: list[NDArray[np.uint8]]) -> float:
        grayscale = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        brightness = float(np.mean([float(frame.mean()) for frame in grayscale]))
        contrast = float(np.mean([float(frame.std()) for frame in grayscale]))
        visible_fraction = float(
            np.mean([float(np.count_nonzero(frame > 5)) / frame.size for frame in grayscale])
        )
        motion = 0.0
        if len(grayscale) > 1:
            motion = float(
                np.mean(
                    [
                        float(cv2.absdiff(previous, current).mean())
                        for previous, current in zip(grayscale, grayscale[1:])
                    ]
                )
            )
        return brightness + 0.5 * contrast + 5.0 * visible_fraction + motion

    def _parse_manual_device(self, value: str) -> str | int:
        if value.isdigit():
            return int(value)
        if sys.platform.startswith("linux") and value.startswith("/dev/"):
            index = self._linux_index_from_path(Path(value))
            if index is not None:
                return index
        return value

    @staticmethod
    def _linux_index_from_path(path: Path) -> int | None:
        try:
            resolved = Path(os.path.realpath(path))
        except OSError:
            resolved = path
        name = resolved.name
        if not name.startswith("video"):
            return None
        suffix = name.removeprefix("video")
        return int(suffix) if suffix.isdigit() else None

    def _video_path_sort_key(self, path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("video")
        if suffix.isdigit():
            return int(suffix), path.name
        return self._maximum_index + 1, path.name
