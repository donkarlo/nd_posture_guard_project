from __future__ import annotations

from dataclasses import dataclass, replace


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PostureGeometry:
    """Seven image-space landmarks defining the face triangle and shoulder lines."""

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
        """Canonicalize clicks by image position so mirror/anatomical left-right cannot swap semantics."""
        if len(points) != 7:
            raise ValueError("Exactly seven geometry points are required.")

        eye_a, eye_b, chin, shoulder_a_1, shoulder_a_2, shoulder_b_1, shoulder_b_2 = points
        left_eye, right_eye = sorted((eye_a, eye_b), key=lambda point: (point[0], point[1]))

        segment_a = (shoulder_a_1, shoulder_a_2)
        segment_b = (shoulder_b_1, shoulder_b_2)
        left_segment, right_segment = sorted(
            (segment_a, segment_b),
            key=lambda segment: (
                0.5 * (segment[0][0] + segment[1][0]),
                0.5 * (segment[0][1] + segment[1][1]),
            ),
        )

        left_outer, left_inner = sorted(
            left_segment,
            key=lambda point: (point[0], point[1]),
        )
        right_inner, right_outer = sorted(
            right_segment,
            key=lambda point: (point[0], point[1]),
        )

        return cls(
            left_eye=left_eye,
            right_eye=right_eye,
            chin=chin,
            left_shoulder_inner=left_inner,
            left_shoulder_outer=left_outer,
            right_shoulder_inner=right_inner,
            right_shoulder_outer=right_outer,
            tracking_confidence=float(tracking_confidence),
        )

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
