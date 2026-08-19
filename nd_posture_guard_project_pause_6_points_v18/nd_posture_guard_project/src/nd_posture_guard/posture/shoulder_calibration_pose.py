from __future__ import annotations

from dataclasses import dataclass

from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


@dataclass(frozen=True, slots=True)
class ShoulderCalibrationPose:
    left_points: tuple[tuple[float, float], ...]
    right_points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.left_points) < 2 or len(self.right_points) < 2:
            raise ValueError("Each shoulder needs at least two calibration points.")
        if len(self.left_points) != len(self.right_points):
            raise ValueError("Both shoulders must use the same number of calibration points.")

    @property
    def points_per_shoulder(self) -> int:
        return len(self.left_points)

    @property
    def all_points(self) -> tuple[tuple[float, float], ...]:
        return self.left_points + self.right_points

    @property
    def shoulders(self) -> TrackedShoulders:
        left = (
            sum(point[0] for point in self.left_points) / len(self.left_points),
            sum(point[1] for point in self.left_points) / len(self.left_points),
        )
        right = (
            sum(point[0] for point in self.right_points) / len(self.right_points),
            sum(point[1] for point in self.right_points) / len(self.right_points),
        )
        return TrackedShoulders(
            left=left,
            right=right,
            anchors=self.all_points,
            valid_anchors=(True,) * len(self.all_points),
        )
