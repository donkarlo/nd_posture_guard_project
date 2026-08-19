from __future__ import annotations

from dataclasses import dataclass

from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


@dataclass(frozen=True, slots=True)
class ShoulderCalibrationPose:
    left_points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    right_points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    @property
    def all_points(self) -> tuple[tuple[float, float], ...]:
        return self.left_points + self.right_points

    @property
    def shoulders(self) -> TrackedShoulders:
        left = (
            sum(point[0] for point in self.left_points) / 3.0,
            sum(point[1] for point in self.left_points) / 3.0,
        )
        right = (
            sum(point[0] for point in self.right_points) / 3.0,
            sum(point[1] for point in self.right_points) / 3.0,
        )
        return TrackedShoulders(
            left=left,
            right=right,
            anchors=self.all_points,
            valid_anchors=(True, True, True, True, True, True),
        )
