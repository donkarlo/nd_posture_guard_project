from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.training.shoulder_roi_builder import ShoulderRoiBuilder


class ExampleTrainingSession:
    """Collect one append-only GOOD or BAD training example."""

    POINTS_PER_SAMPLE = 6

    def __init__(self, roi_builder: ShoulderRoiBuilder, frames_per_sample: int) -> None:
        self._roi_builder = roi_builder
        self._frames_per_sample = max(8, int(frames_per_sample))
        self.clear()

    @property
    def active(self) -> bool:
        return self._phase != "idle"

    @property
    def waiting_for_frame(self) -> bool:
        return self._phase == "waiting_frame"

    @property
    def selecting_points(self) -> bool:
        return self._phase == "selecting"

    @property
    def recording(self) -> bool:
        return self._phase == "recording"

    @property
    def complete(self) -> bool:
        return self._phase == "complete"

    @property
    def label(self) -> int:
        return self._label

    @property
    def label_name(self) -> str:
        return "GOOD" if self._label == 0 else "BAD"

    @property
    def frames_per_sample(self) -> int:
        return self._frames_per_sample

    @property
    def recording_count(self) -> int:
        return len(self._features)

    @property
    def current_points(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._points)

    @property
    def next_point_message(self) -> str:
        index = len(self._points)
        side = "LEFT" if index < 3 else "RIGHT"
        local = index if index < 3 else index - 3
        descriptions = ("neck-side/inner", "middle", "outer shoulder tip")
        return (
            f"{self.label_name} example — click {side} shoulder point {local + 1}/3: "
            f"{descriptions[local]}. Use the visible top shoulder line."
        )

    @property
    def left_roi(self) -> tuple[int, int, int, int] | None:
        return self._left_roi

    @property
    def right_roi(self) -> tuple[int, int, int, int] | None:
        return self._right_roi

    @property
    def anchor_frame(self) -> NDArray[np.uint8]:
        if self._anchor_frame is None:
            raise RuntimeError("Training anchor frame is not available.")
        return self._anchor_frame

    @property
    def features(self) -> NDArray[np.float32]:
        if not self._features:
            raise RuntimeError("Training features are not available.")
        return np.vstack(self._features).astype(np.float32)

    @property
    def frames(self) -> tuple[NDArray[np.uint8], ...]:
        return tuple(self._frames)

    def start(self, label: int) -> None:
        if label not in (0, 1):
            raise ValueError("Training label must be 0 or 1.")
        self.clear()
        self._label = int(label)
        self._phase = "waiting_frame"

    def begin_point_selection(self, frame: NDArray[np.uint8]) -> None:
        if not self.waiting_for_frame:
            raise RuntimeError("Training is not waiting for a frame.")
        self._anchor_frame = frame.copy()
        self._phase = "selecting"

    def add_point(self, point: tuple[float, float], frame_width: int, frame_height: int) -> bool:
        if not self.selecting_points:
            return False
        x, y = point
        if not (0 <= x < frame_width and 0 <= y < frame_height):
            return False
        self._points.append((float(x), float(y)))
        if len(self._points) < self.POINTS_PER_SAMPLE:
            return False
        self._left_roi, self._right_roi = self._roi_builder.build(
            tuple(self._points), frame_width, frame_height
        )
        self._phase = "recording"
        return True

    def add_observation(self, feature: NDArray[np.float32], frame: NDArray[np.uint8]) -> bool:
        if not self.recording:
            return False
        self._features.append(np.asarray(feature, dtype=np.float32))
        self._frames.append(frame.copy())
        if len(self._features) < self._frames_per_sample:
            return False
        self._phase = "complete"
        return True

    def clear(self) -> None:
        self._phase = "idle"
        self._label = 0
        self._points: list[tuple[float, float]] = []
        self._left_roi: tuple[int, int, int, int] | None = None
        self._right_roi: tuple[int, int, int, int] | None = None
        self._anchor_frame: NDArray[np.uint8] | None = None
        self._features: list[NDArray[np.float32]] = []
        self._frames: list[NDArray[np.uint8]] = []
