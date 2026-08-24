from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.training.example_training_session import ExampleTrainingSession
from nd_posture_guard.vision.mediapipe_posture_geometry_detector import (
    MediaPipePostureGeometryDetector,
)


class AutoDetectedTrainingSession(ExampleTrainingSession):
    """Collect a GOOD/BAD sample from fresh MediaPipe detections; no clicks."""

    def __init__(
        self,
        frames_per_sample: int,
        detector: MediaPipePostureGeometryDetector,
    ) -> None:
        self._detector = detector
        self._detected_point_sets: list[tuple[tuple[float, float], ...]] = []
        self._attempt_count = 0
        super().__init__(frames_per_sample)

    @property
    def next_point_message(self) -> str:
        return (
            f"{self.label_name} automatic landmark capture — hold this posture. "
            "Eyes, chin and both shoulder lines are detected independently on every frame."
        )

    def start(self, label: int) -> None:
        if label not in (0, 1):
            raise ValueError("Training label must be 0 or 1.")
        self.clear()
        self._label = int(label)
        self._phase = "recording"

    def begin_point_selection(self, frame: NDArray[np.uint8]) -> None:
        raise RuntimeError("Automatic landmark training does not use point selection.")

    def add_point(
        self,
        point: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        return False

    def add_frame(self, frame: NDArray[np.uint8]) -> bool:
        if not self.recording:
            return False

        self._attempt_count += 1
        geometry = self._detector.detect(frame)
        if geometry is None or geometry.tracking_confidence < 0.50:
            return False

        self._frames.append(frame.copy())
        self._detected_point_sets.append(geometry.points)
        if len(self._frames) < self._frames_per_sample:
            return False

        point_array = np.asarray(self._detected_point_sets, dtype=np.float64)
        median_points = np.median(point_array, axis=0)
        median_geometry = PostureGeometry.from_points(
            tuple((float(point[0]), float(point[1])) for point in median_points)
        )

        flattened = point_array.reshape(point_array.shape[0], -1)
        median_flat = median_points.reshape(-1)
        representative_index = int(
            np.argmin(np.linalg.norm(flattened - median_flat, axis=1))
        )
        self._anchor_frame = self._frames[representative_index].copy()
        self._points = list(median_geometry.points)
        self._phase = "complete"
        return True

    def clear(self) -> None:
        super().clear()
        self._detected_point_sets = []
        self._attempt_count = 0
