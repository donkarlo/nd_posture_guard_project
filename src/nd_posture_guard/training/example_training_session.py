from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import PostureGeometry


class ExampleTrainingSession:
    """Collect one GOOD or BAD sample as three drawings made from seven landmarks."""

    POINTS_PER_SAMPLE = 7
    SHAPES_PER_SAMPLE = 3
    POINT_MESSAGES = (
        "FACE TRIANGLE — center of either eye (image position is canonicalized automatically)",
        "FACE TRIANGLE — center of the other eye",
        "FACE TRIANGLE — lowest visible point of the chin",
        "SHOULDER LINE A — one endpoint on either shoulder line",
        "SHOULDER LINE A — the other endpoint of that same shoulder line",
        "SHOULDER LINE B — one endpoint on the shoulder line on the other side of the image",
        "SHOULDER LINE B — the other endpoint of that same shoulder line",
    )

    def __init__(self, frames_per_sample: int) -> None:
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
        return len(self._frames)

    @property
    def current_points(self) -> tuple[tuple[float, float], ...]:
        """Return canonical image-space landmarks once all seven clicks are available."""
        points = tuple(self._points)
        if len(points) == self.POINTS_PER_SAMPLE:
            return PostureGeometry.from_points(points).points
        return points

    @property
    def geometry(self) -> PostureGeometry:
        if len(self._points) != self.POINTS_PER_SAMPLE:
            raise RuntimeError("Training geometry is not complete.")
        return PostureGeometry.from_points(tuple(self._points))

    @property
    def next_point_message(self) -> str:
        index = min(len(self._points), self.POINTS_PER_SAMPLE - 1)
        shape_number, point_number, points_in_shape = self._shape_progress(index)
        return (
            f"{self.label_name} example — SHAPE {shape_number}/{self.SHAPES_PER_SAMPLE}, "
            f"point {point_number}/{points_in_shape}: click the {self.POINT_MESSAGES[index]}. "
            "The program determines LEFT SIDE OF IMAGE / RIGHT SIDE OF IMAGE and inner/outer shoulder endpoints automatically."
        )

    @property
    def anchor_frame(self) -> NDArray[np.uint8]:
        if self._anchor_frame is None:
            raise RuntimeError("Training anchor frame is not available.")
        return self._anchor_frame

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
        """Append one click; canonicalize eye/shoulder left-right and endpoint order before validation."""
        if not self.selecting_points:
            return False
        x, y = point
        if not (0 <= x < frame_width and 0 <= y < frame_height):
            raise ValueError("The selected point is outside the camera image.")
        self._points.append((float(x), float(y)))
        if len(self._points) < self.POINTS_PER_SAMPLE:
            return False

        geometry = PostureGeometry.from_points(tuple(self._points))
        if not geometry.is_plausible(frame_width, frame_height):
            self._points.clear()
            raise ValueError(
                "The three drawings are not plausible even after automatic image-side normalization. "
                "Redraw the eye-eye-chin triangle and one complete line on each shoulder."
            )
        self._points = list(geometry.points)
        self._phase = "recording"
        return True

    def add_frame(self, frame: NDArray[np.uint8]) -> bool:
        if not self.recording:
            return False
        self._frames.append(frame.copy())
        if len(self._frames) < self._frames_per_sample:
            return False
        self._phase = "complete"
        return True

    def clear(self) -> None:
        self._phase = "idle"
        self._label = 0
        self._points: list[tuple[float, float]] = []
        self._anchor_frame: NDArray[np.uint8] | None = None
        self._frames: list[NDArray[np.uint8]] = []

    @staticmethod
    def _shape_progress(point_index: int) -> tuple[int, int, int]:
        if point_index < 3:
            return 1, point_index + 1, 3
        if point_index < 5:
            return 2, point_index - 2, 2
        return 3, point_index - 4, 2
