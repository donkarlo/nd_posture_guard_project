from __future__ import annotations

from dataclasses import dataclass
import math

from nd_posture_guard.vision.tracked_shoulders import TrackedShoulders


@dataclass(frozen=True, slots=True)
class ShoulderCalibrationProfile:
    center_y_ratio: float
    width_ratio: float
    tilt_degrees: float

    @classmethod
    def from_shoulders(
        cls,
        shoulders: TrackedShoulders,
        frame_width: int,
        frame_height: int,
    ) -> "ShoulderCalibrationProfile":
        width = max(float(frame_width), 1.0)
        height = max(float(frame_height), 1.0)
        dx = shoulders.right[0] - shoulders.left[0]
        dy = shoulders.right[1] - shoulders.left[1]
        return cls(
            center_y_ratio=shoulders.center[1] / height,
            width_ratio=shoulders.width / width,
            tilt_degrees=math.degrees(math.atan2(dy, dx)),
        )
