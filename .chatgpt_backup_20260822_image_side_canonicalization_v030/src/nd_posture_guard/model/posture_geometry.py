from __future__ import annotations

from dataclasses import dataclass, replace


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PostureGeometry:
    """Seven tracked landmarks that define the face triangle and shoulder lines."""

    left_eye: Point
    right_eye: Point
    chin: Point
    left_shoulder_inner: Point
    left_shoulder_outer: Point
    right_shoulder_inner: Point
    right_shoulder_outer: Point
    tracking_confidence: float = 1.0

    @classmethod
    def from_points(
        cls,
        points: tuple[Point, ...],
        tracking_confidence: float = 1.0,
    ) -> "PostureGeometry":
        if len(points) != 7:
            raise ValueError("Exactly seven geometry points are required.")
        return cls(*points, tracking_confidence=float(tracking_confidence))

    @property
    def points(self) -> tuple[Point, ...]:
        return (
            self.left_eye,
            self.right_eye,
            self.chin,
            self.left_shoulder_inner,
            self.left_shoulder_outer,
            self.right_shoulder_inner,
            self.right_shoulder_outer,
        )

    def with_confidence(self, value: float) -> "PostureGeometry":
        return replace(self, tracking_confidence=float(value))

    def is_plausible(self, frame_width: int, frame_height: int) -> bool:
        """Reject geometries that violate basic screen-space body ordering."""
        width = max(1, int(frame_width))
        height = max(1, int(frame_height))
        margin_x = width * 0.03
        margin_y = height * 0.03
        for x, y in self.points:
            if not (-margin_x <= x <= width + margin_x and -margin_y <= y <= height + margin_y):
                return False

        if self.left_eye[0] >= self.right_eye[0]:
            return False
        if self.chin[1] <= max(self.left_eye[1], self.right_eye[1]):
            return False
        if self.left_shoulder_outer[0] >= self.left_shoulder_inner[0]:
            return False
        if self.right_shoulder_inner[0] >= self.right_shoulder_outer[0]:
            return False
        if self.left_shoulder_inner[0] >= self.right_shoulder_inner[0]:
            return False
        if self.left_shoulder_outer[0] >= self.right_shoulder_outer[0]:
            return False

        shoulder_span = self.right_shoulder_outer[0] - self.left_shoulder_outer[0]
        if shoulder_span < width * 0.18:
            return False

        eye_span = self.right_eye[0] - self.left_eye[0]
        if eye_span < width * 0.025 or eye_span > shoulder_span * 0.75:
            return False

        inner_y = 0.5 * (self.left_shoulder_inner[1] + self.right_shoulder_inner[1])
        if self.chin[1] > inner_y + height * 0.12:
            return False
        return True
