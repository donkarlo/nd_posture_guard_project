from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraDeviceFinder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        maximum_index: int = 9,
        test_frame_count: int = 6,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._maximum_index = maximum_index
        self._test_frame_count = test_frame_count

    def find(self, configured_device: str) -> str | int:
        normalized = configured_device.strip()
        if normalized.lower() != "auto":
            return self._parse_manual_device(normalized)

        candidates = self._candidate_devices()
        best_device: str | int | None = None
        best_score = float("-inf")

        for candidate in candidates:
            score = self._score_device(candidate)
            if score is not None and score > best_score:
                best_device = candidate
                best_score = score

        if best_device is None:
            raise RuntimeError(
                "No usable webcam was found automatically. "
                "Check that a webcam is connected and that the user has camera permission."
            )

        return best_device

    def _candidate_devices(self) -> list[str | int]:
        if sys.platform.startswith("linux"):
            linux_candidates = self._linux_candidates()
            if linux_candidates:
                return linux_candidates
        return list(range(self._maximum_index + 1))

    def _linux_candidates(self) -> list[str]:
        candidates: list[str] = []
        seen_targets: set[str] = set()

        by_id_directory = Path("/dev/v4l/by-id")
        if by_id_directory.is_dir():
            for path in sorted(by_id_directory.glob("*-video-index0")):
                self._append_linux_candidate(path, candidates, seen_targets)

        for path in sorted(Path("/dev").glob("video*"), key=self._video_path_sort_key):
            self._append_linux_candidate(path, candidates, seen_targets)

        return candidates

    def _append_linux_candidate(
        self,
        path: Path,
        candidates: list[str],
        seen_targets: set[str],
    ) -> None:
        try:
            target = os.path.realpath(path)
        except OSError:
            target = str(path)

        if target in seen_targets:
            return

        seen_targets.add(target)
        candidates.append(str(path))

    def _score_device(self, device: str | int) -> float | None:
        capture = self._open_capture(device)
        if not capture.isOpened():
            capture.release()
            return None

        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            capture.set(cv2.CAP_PROP_FPS, self._fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        if isinstance(device, str) and device.startswith("/dev/"):
            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(device)

    def _frame_sequence_score(self, frames: list[NDArray[np.uint8]]) -> float:
        grayscale_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        brightness = float(np.mean([float(frame.mean()) for frame in grayscale_frames]))
        contrast = float(np.mean([float(frame.std()) for frame in grayscale_frames]))
        visible_fraction = float(
            np.mean([float(np.count_nonzero(frame > 5)) / frame.size for frame in grayscale_frames])
        )

        motion = 0.0
        if len(grayscale_frames) > 1:
            differences = [
                float(cv2.absdiff(previous, current).mean())
                for previous, current in zip(grayscale_frames, grayscale_frames[1:])
            ]
            motion = float(np.mean(differences))

        return brightness + (0.5 * contrast) + (5.0 * visible_fraction) + motion

    def _parse_manual_device(self, value: str) -> str | int:
        if value.isdigit():
            return int(value)
        return value

    def _video_path_sort_key(self, path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("video")
        if suffix.isdigit():
            return int(suffix), path.name
        return self._maximum_index + 1, path.name
