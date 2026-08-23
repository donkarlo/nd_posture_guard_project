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
        """Accept user-drawn geometry unless it is actually unusable.

        Training geometry is deliberately validated only for hard failures. Body
        proportions vary with pose, camera perspective and framing, so semantic
        assumptions such as minimum shoulder span or chin-to-shoulder ordering must
        not reject an otherwise valid sample.
        """
        width = max(1, int(frame_width))
        height = max(1, int(frame_height))
        margin_x = width * 0.03
        margin_y = height * 0.03

        for x, y in self.points:
            if not (
                -margin_x <= x <= width + margin_x
                and -margin_y <= y <= height + margin_y
            ):
                return False

        minimum_squared_length = 4.0

        def squared_distance(point_a: Point, point_b: Point) -> float:
            delta_x = point_b[0] - point_a[0]
            delta_y = point_b[1] - point_a[1]
            return delta_x * delta_x + delta_y * delta_y

        if squared_distance(self.left_eye, self.right_eye) <= minimum_squared_length:
            return False

        triangle_twice_area = abs(
            (self.right_eye[0] - self.left_eye[0])
            * (self.chin[1] - self.left_eye[1])
            - (self.right_eye[1] - self.left_eye[1])
            * (self.chin[0] - self.left_eye[0])
        )
        if triangle_twice_area <= 4.0:
            return False

        if (
            squared_distance(self.left_shoulder_inner, self.left_shoulder_outer)
            <= minimum_squared_length
        ):
            return False

        if (
            squared_distance(self.right_shoulder_inner, self.right_shoulder_outer)
            <= minimum_squared_length
        ):
            return False

        return True
