from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import PostureGeometry
from nd_posture_guard.vision.posture_geometry_tracker import PostureGeometryTracker


class FastPostureGeometryTracker(PostureGeometryTracker):
    """Track the seven learned landmarks with optical flow only.

    The original tracker performs many template matches after each profile reset
    and periodically while monitoring. This version seeds from the learned
    normalized geometry immediately and uses LK optical flow, avoiding long
    matchTemplate bursts in the monitoring loop.
    """

    INITIAL_CONFIDENCE = 0.80

    def track(self, frame: NDArray[np.uint8]) -> PostureGeometry | None:
        if self._profile is None:
            return None

        gray = self._to_gray(frame)
        height, width = gray.shape[:2]
        if self._previous_gray is None or self._previous_points is None:
            return self._initialize_from_reference(gray, width, height)

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            self._previous_points,
            None,
            winSize=(23, 23),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 0.02),
        )
        if next_points is None or status is None:
            return self._initialize_from_reference(gray, width, height)

        points = next_points.reshape(-1, 2).astype(np.float32)
        previous = self._previous_points.reshape(-1, 2).astype(np.float32)
        status_values = status.reshape(-1)
        error_values = (
            errors.reshape(-1)
            if errors is not None
            else np.full(self.POINT_COUNT, 99.0, dtype=np.float32)
        )
        if points.shape != (self.POINT_COUNT, 2):
            return self._initialize_from_reference(gray, width, height)

        scores = np.zeros(self.POINT_COUNT, dtype=np.float32)
        valid_count = 0
        for index in range(self.POINT_COUNT):
            point = points[index]
            displacement = float(np.linalg.norm(point - previous[index]))
            valid = (
                bool(status_values[index])
                and np.all(np.isfinite(point))
                and float(error_values[index]) <= 40.0
                and displacement <= 90.0
                and -10.0 <= float(point[0]) <= width + 10.0
                and -10.0 <= float(point[1]) <= height + 10.0
            )
            if valid:
                valid_count += 1
                scores[index] = float(max(0.35, 1.0 - float(error_values[index]) / 55.0))
            else:
                points[index] = previous[index]
                scores[index] = 0.10

        if valid_count < 5:
            return self._initialize_from_reference(gray, width, height)

        self._previous_gray = gray
        self._previous_points = points.reshape(-1, 1, 2).astype(np.float32)
        self._frame_counter += 1
        tuples = tuple((float(point[0]), float(point[1])) for point in points)
        confidence = float(np.clip(np.median(scores), 0.0, 1.0))
        geometry = PostureGeometry.from_points(tuples, confidence)
        if geometry.is_plausible(width, height):
            self._last_good_geometry = geometry
            return geometry
        if self._last_good_geometry is not None:
            return self._last_good_geometry.with_confidence(0.0)
        return self._initialize_from_reference(gray, width, height)

    def _initialize_from_reference(
        self,
        gray: NDArray[np.uint8],
        width: int,
        height: int,
    ) -> PostureGeometry | None:
        assert self._profile is not None
        points = np.asarray(
            [
                (float(nx) * width, float(ny) * height)
                for nx, ny in self._profile.reference_points_normalized
            ],
            dtype=np.float32,
        )
        if points.shape != (self.POINT_COUNT, 2):
            return None
        self._previous_gray = gray
        self._previous_points = points.reshape(-1, 1, 2)
        self._frame_counter = 1
        geometry = PostureGeometry.from_points(
            tuple((float(point[0]), float(point[1])) for point in points),
            self.INITIAL_CONFIDENCE,
        )
        if not geometry.is_plausible(width, height):
            self._previous_points = None
            return None
        self._last_good_geometry = geometry
        return geometry
