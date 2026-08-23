from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from nd_posture_guard.model.posture_geometry import Point, PostureGeometry
from nd_posture_guard.training.posture_training_profile import PostureTrainingProfile


class PostureGeometryTracker:
    """Track the seven learned posture landmarks with optical flow and template re-anchoring."""

    POINT_COUNT = 7

    def __init__(
        self,
        initial_search_radius: int = 70,
        recovery_search_radius: int = 45,
        refine_search_radius: int = 16,
        reanchor_every_frames: int = 12,
        minimum_template_score: float = 0.24,
    ) -> None:
        self._initial_search_radius = max(24, int(initial_search_radius))
        self._recovery_search_radius = max(20, int(recovery_search_radius))
        self._refine_search_radius = max(8, int(refine_search_radius))
        self._reanchor_every_frames = max(4, int(reanchor_every_frames))
        self._minimum_template_score = float(minimum_template_score)
        self._profile: PostureTrainingProfile | None = None
        self.reset()

    def set_profile(self, profile: PostureTrainingProfile | None) -> None:
        self._profile = profile
        self.reset()

    def reset(self) -> None:
        self._previous_gray: NDArray[np.uint8] | None = None
        self._previous_points: NDArray[np.float32] | None = None
        self._last_good_geometry: PostureGeometry | None = None
        self._frame_counter = 0

    def track(self, frame: NDArray[np.uint8]) -> PostureGeometry | None:
        if self._profile is None:
            return None
        gray = self._to_gray(frame)
        height, width = gray.shape[:2]

        if self._previous_gray is None or self._previous_points is None:
            points, scores = self._initialize(gray, width, height)
        else:
            points, scores = self._follow_and_refine(gray, width, height)

        self._frame_counter += 1
        geometry = self._geometry_from_points(points, scores, width, height)
        self._previous_gray = gray
        self._previous_points = points.reshape(-1, 1, 2).astype(np.float32)

        if geometry is not None and geometry.is_plausible(width, height):
            self._last_good_geometry = geometry
            return geometry

        # Do not classify from an implausible geometry. Keeping a low-confidence
        # last geometry lets the UI still show the best-known triangle/lines.
        if self._last_good_geometry is not None:
            return self._last_good_geometry.with_confidence(0.0)
        return None

    def _initialize(
        self,
        gray: NDArray[np.uint8],
        width: int,
        height: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert self._profile is not None
        points: list[Point] = []
        scores: list[float] = []
        for index, normalized in enumerate(self._profile.reference_points_normalized):
            expected = (normalized[0] * width, normalized[1] * height)
            point, score = self._match_point(
                gray,
                expected,
                self._profile.point_templates[index],
                self._initial_search_radius,
            )
            points.append(point)
            scores.append(score)
        return np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32)

    def _follow_and_refine(
        self,
        gray: NDArray[np.uint8],
        width: int,
        height: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert self._profile is not None
        assert self._previous_gray is not None
        assert self._previous_points is not None

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            gray,
            self._previous_points,
            None,
            winSize=(23, 23),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                18,
                0.02,
            ),
        )
        if next_points is None or status is None:
            return self._initialize(gray, width, height)

        points = next_points.reshape(-1, 2).astype(np.float32)
        status_values = status.reshape(-1)
        error_values = (
            errors.reshape(-1) if errors is not None else np.full(self.POINT_COUNT, 50.0)
        )
        previous = self._previous_points.reshape(-1, 2)
        scores = np.zeros(self.POINT_COUNT, dtype=np.float32)
        refine_all = self._frame_counter % self._reanchor_every_frames == 0

        for index in range(self.POINT_COUNT):
            displacement = float(np.linalg.norm(points[index] - previous[index]))
            flow_ok = (
                bool(status_values[index])
                and np.all(np.isfinite(points[index]))
                and float(error_values[index]) <= 35.0
                and displacement <= self._recovery_search_radius * 1.8
                and -8.0 <= points[index, 0] <= width + 8.0
                and -8.0 <= points[index, 1] <= height + 8.0
            )
            if not flow_ok:
                recovered, score = self._match_point(
                    gray,
                    (float(previous[index, 0]), float(previous[index, 1])),
                    self._profile.point_templates[index],
                    self._recovery_search_radius,
                )
                points[index] = recovered
                scores[index] = score
                continue

            # Convert LK error to a bounded confidence. This is intentionally
            # conservative; periodic template matching corrects slow drift.
            scores[index] = float(max(0.30, 1.0 - float(error_values[index]) / 45.0))
            if refine_all:
                refined, template_score = self._match_point(
                    gray,
                    (float(points[index, 0]), float(points[index, 1])),
                    self._profile.point_templates[index],
                    self._refine_search_radius,
                )
                if template_score >= self._minimum_template_score:
                    points[index] = refined
                    scores[index] = max(scores[index], template_score)

        return points, scores

    def _match_point(
        self,
        gray: NDArray[np.uint8],
        expected: Point,
        templates: tuple[NDArray[np.uint8], ...],
        radius: int,
    ) -> tuple[Point, float]:
        height, width = gray.shape[:2]
        ex, ey = expected
        best_point = (
            float(min(max(ex, 0.0), width - 1.0)),
            float(min(max(ey, 0.0), height - 1.0)),
        )
        best_score = -1.0

        for template in templates:
            if template.ndim != 2 or template.size == 0:
                continue
            th, tw = template.shape[:2]
            half_w = tw // 2
            half_h = th // 2
            x1 = max(0, int(round(ex - radius - half_w)))
            y1 = max(0, int(round(ey - radius - half_h)))
            x2 = min(width, int(round(ex + radius + half_w + 1)))
            y2 = min(height, int(round(ey + radius + half_h + 1)))
            search = gray[y1:y2, x1:x2]
            if search.shape[0] < th or search.shape[1] < tw:
                continue
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            if not np.isfinite(score) or float(score) <= best_score:
                continue
            best_score = float(score)
            best_point = (
                float(x1 + location[0] + half_w),
                float(y1 + location[1] + half_h),
            )

        return best_point, max(0.0, best_score)

    @staticmethod
    def _geometry_from_points(
        points: NDArray[np.float32],
        scores: NDArray[np.float32],
        width: int,
        height: int,
    ) -> PostureGeometry | None:
        if points.shape != (7, 2) or scores.shape != (7,):
            return None
        if not np.all(np.isfinite(points)):
            return None
        tuples = tuple((float(point[0]), float(point[1])) for point in points)
        confidence = float(np.clip(np.median(scores), 0.0, 1.0))
        geometry = PostureGeometry.from_points(tuples, confidence)
        if not geometry.is_plausible(width, height):
            return None
        return geometry

    @staticmethod
    def _to_gray(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise RuntimeError("Unexpected camera image format.")
        return cv2.GaussianBlur(gray, (3, 3), 0)
