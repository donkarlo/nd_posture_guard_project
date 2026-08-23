from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import PostureGeometry


class ExampleTrainingSession:
    """Collect one GOOD or BAD sample as three drawings made from seven landmarks."""

    POINTS_PER_SAMPLE = 7
    SHAPES_PER_SAMPLE = 3
    POINT_MESSAGES = (
        "FACE TRIANGLE — center of the eye on the LEFT side of the image",
        "FACE TRIANGLE — center of the eye on the RIGHT side of the image",
        "FACE TRIANGLE — lowest visible point of the chin",
        "LEFT SHOULDER LINE — neck/shoulder junction",
        "LEFT SHOULDER LINE — outer end of the shoulder",
        "RIGHT SHOULDER LINE — neck/shoulder junction",
        "RIGHT SHOULDER LINE — outer end of the shoulder",
    )

    def __init__(self, frames_per_sample: int) -> None:
        self._frames_per_sample = max(8, int(frames_per_sample))
        self.clear()

    @property
    def active(self) -> bool:
        """Return whether a sample is currently being collected."""
        return self._phase != "idle"

    @property
    def waiting_for_frame(self) -> bool:
        """Return whether the session is waiting for the frozen click frame."""
        return self._phase == "waiting_frame"

    @property
    def selecting_points(self) -> bool:
        """Return whether the user is currently drawing the three training shapes."""
        return self._phase == "selecting"

    @property
    def recording(self) -> bool:
        """Return whether the short review clip is being captured."""
        return self._phase == "recording"

    @property
    def complete(self) -> bool:
        """Return whether all requested recording frames were collected."""
        return self._phase == "complete"

    @property
    def label(self) -> int:
        """Return the numeric training label."""
        return self._label

    @property
    def label_name(self) -> str:
        """Return the human-readable training class name."""
        return "GOOD" if self._label == 0 else "BAD"

    @property
    def frames_per_sample(self) -> int:
        """Return the requested review-video frame count."""
        return self._frames_per_sample

    @property
    def recording_count(self) -> int:
        """Return how many review-video frames have been captured."""
        return len(self._frames)

    @property
    def current_points(self) -> tuple[tuple[float, float], ...]:
        """Return the currently selected landmarks in semantic order."""
        return tuple(self._points)

    @property
    def geometry(self) -> PostureGeometry:
        """Return the completed face triangle plus both shoulder segments."""
        if len(self._points) != self.POINTS_PER_SAMPLE:
            raise RuntimeError("Training geometry is not complete.")
        return PostureGeometry.from_points(tuple(self._points))

    @property
    def next_point_message(self) -> str:
        """Describe the current drawing and the exact next defining point."""
        index = min(len(self._points), self.POINTS_PER_SAMPLE - 1)
        shape_number, point_number, points_in_shape = self._shape_progress(index)
        return (
            f"{self.label_name} example — SHAPE {shape_number}/{self.SHAPES_PER_SAMPLE}, "
            f"point {point_number}/{points_in_shape}: click the {self.POINT_MESSAGES[index]}. "
            f"Shape 1 is the eye-eye-chin triangle; shapes 2 and 3 are the left and right shoulder lines."
        )

    @property
    def anchor_frame(self) -> NDArray[np.uint8]:
        """Return the frozen frame on which geometry was manually defined."""
        if self._anchor_frame is None:
            raise RuntimeError("Training anchor frame is not available.")
        return self._anchor_frame

    @property
    def frames(self) -> tuple[NDArray[np.uint8], ...]:
        """Return the short raw clip retained for review and future migration."""
        return tuple(self._frames)

    def start(self, label: int) -> None:
        """Start a fresh GOOD or BAD sample."""
        if label not in (0, 1):
            raise ValueError("Training label must be 0 or 1.")
        self.clear()
        self._label = int(label)
        self._phase = "waiting_frame"

    def begin_point_selection(self, frame: NDArray[np.uint8]) -> None:
        """Freeze the anchor image and begin the three-shape drawing sequence."""
        if not self.waiting_for_frame:
            raise RuntimeError("Training is not waiting for a frame.")
        self._anchor_frame = frame.copy()
        self._phase = "selecting"

    def add_point(self, point: tuple[float, float], frame_width: int, frame_height: int) -> bool:
        """Append one defining point and validate all three completed shapes."""
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
                "The three drawings are not plausible. Redraw the eye-eye-chin triangle, "
                "then the left shoulder line from neck junction to shoulder end, then the right shoulder line."
            )
        self._phase = "recording"
        return True

    def add_frame(self, frame: NDArray[np.uint8]) -> bool:
        """Store one review-video frame and report when the sample is complete."""
        if not self.recording:
            return False
        self._frames.append(frame.copy())
        if len(self._frames) < self._frames_per_sample:
            return False
        self._phase = "complete"
        return True

    def clear(self) -> None:
        """Reset all transient training state and release captured frame references."""
        self._phase = "idle"
        self._label = 0
        self._points: list[tuple[float, float]] = []
        self._anchor_frame: NDArray[np.uint8] | None = None
        self._frames: list[NDArray[np.uint8]] = []

    @staticmethod
    def _shape_progress(point_index: int) -> tuple[int, int, int]:
        """Map the seven-point sequence to three user-visible drawings."""
        if point_index < 3:
            return 1, point_index + 1, 3
        if point_index < 5:
            return 2, point_index - 2, 2
        return 3, point_index - 4, 2
